"""
Main DICOM to MHA converter.

This module orchestrates the full conversion pipeline from DICOM files
exported from Mimics to MHA files compatible with nnU-Net training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from .dicom_reader import DicomSeriesReader
from .label_combiner import LabelCombiner, OverlapInfo
from .mask_processor import MimicsMaskProcessor
from .nnunet_formatter import NnunetFormatter
from .validator import (
    SegmentationValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


@dataclass
class ConversionConfig:
    """Configuration for DICOM to MHA conversion."""

    # Input/Output
    input_dir: Path
    output_dir: Path
    image_folder_name: str  # Name of the folder containing the main DICOM image

    # Mask processing
    hu_tolerance: float = 0.5  # Tolerance for uniform HU detection
    min_mask_size: int = 100  # Minimum voxels for valid mask

    # nnU-Net formatting
    dataset_name: str = "KneeSegmentation"
    dataset_id: int = 1
    modality: str = "CT"  # or "MR"

    # Validation
    strict_validation: bool = False  # If True, fail on overlaps; if False, just warn

    # Output options
    compress: bool = True  # Use compression in MHA


@dataclass
class MaskInfo:
    """Information about a processed mask."""

    folder_name: str
    label_value: int
    detected_hu: float
    voxel_count: int
    volume_mm3: float


@dataclass
class ConversionResult:
    """Result of converting a single patient."""

    patient_id: str
    success: bool = False
    error_message: Optional[str] = None

    # Image info
    image_shape: Optional[tuple] = None
    image_spacing: Optional[tuple] = None

    # Mask info
    mask_info: list[MaskInfo] = field(default_factory=list)
    label_mapping: Optional[dict[str, int]] = None

    # Validation
    validation_result: Optional[ValidationResult] = None
    overlaps: list[OverlapInfo] = field(default_factory=list)

    # Output paths
    image_path: Optional[Path] = None
    label_path: Optional[Path] = None


class DicomToMhaConverter:
    """Main converter orchestrating the full pipeline."""

    def __init__(self, config: ConversionConfig):
        """
        Initialize the converter.

        Args:
            config: Conversion configuration
        """
        self.config = config
        self.reader = DicomSeriesReader()
        self.mask_processor = MimicsMaskProcessor(
            hu_tolerance=config.hu_tolerance,
            min_region_size=config.min_mask_size,
        )
        self.combiner = LabelCombiner()
        self.validator = SegmentationValidator(strict=config.strict_validation)
        self.formatter = NnunetFormatter(
            dataset_name=config.dataset_name,
            dataset_id=config.dataset_id,
        )

    def convert_patient(self, patient_dir: Path) -> ConversionResult:
        """
        Convert a single patient's DICOM data to nnU-Net format.

        Expected input structure:
        patient_dir/
        ├── <image_folder>/     # Main DICOM series (name from config)
        ├── 00_background/      # Background mask
        ├── 01_femur/           # Mask with label 1
        ├── 02_tibia/           # Mask with label 2
        └── ...

        Args:
            patient_dir: Path to patient directory

        Returns:
            ConversionResult with success status and details
        """
        patient_dir = Path(patient_dir)
        result = ConversionResult(patient_id=patient_dir.name)

        try:
            # Step 1: Identify folders
            image_folder, mask_folders = self._identify_folders(patient_dir)

            if image_folder is None:
                result.error_message = (
                    f"Image folder '{self.config.image_folder_name}' not found in {patient_dir}"
                )
                return result

            print(f"  Processing {patient_dir.name}...")
            print(f"    Image folder: {image_folder.name}")
            print(f"    Mask folders: {[f.name for f in mask_folders]}")

            # Step 2: Read main image DICOM
            image = self.reader.read_dicom_series(image_folder)
            result.image_shape = image.GetSize()
            result.image_spacing = image.GetSpacing()

            # Step 3: Read and process each mask
            processed_masks = {}
            label_mapping = {}

            for mask_folder in mask_folders:
                folder_name = mask_folder.name

                try:
                    # Parse label from folder name
                    label_value, label_name = self.combiner.parse_label_from_folder_name(
                        folder_name
                    )
                except ValueError as e:
                    print(f"    Warning: {e}")
                    continue

                # Read mask DICOM
                try:
                    mask_dicom = self.reader.read_dicom_series(mask_folder)
                except Exception as e:
                    print(f"    Warning: Could not read mask '{folder_name}': {e}")
                    continue

                mask_array = sitk.GetArrayFromImage(mask_dicom)

                # Check dimensions match
                if mask_array.shape != tuple(reversed(image.GetSize())):
                    print(
                        f"    Warning: Mask '{folder_name}' has different dimensions, skipping"
                    )
                    continue

                # Extract uniform HU region (the actual mask)
                binary_mask, detected_hu = self.mask_processor.extract_mask(mask_array)

                # Calculate volume
                voxel_volume = np.prod(image.GetSpacing())
                volume_mm3 = float(np.sum(binary_mask) * voxel_volume)

                result.mask_info.append(
                    MaskInfo(
                        folder_name=folder_name,
                        label_value=label_value,
                        detected_hu=detected_hu,
                        voxel_count=int(np.sum(binary_mask)),
                        volume_mm3=volume_mm3,
                    )
                )

                processed_masks[folder_name] = binary_mask
                label_mapping[folder_name] = label_value

                print(
                    f"    Processed {folder_name}: label={label_value}, "
                    f"HU={detected_hu:.1f}, voxels={np.sum(binary_mask):,}"
                )

            if not processed_masks:
                result.error_message = "No valid masks found"
                return result

            result.label_mapping = label_mapping

            # Step 4: Combine masks into multi-label segmentation
            combine_result = self.combiner.combine(processed_masks, label_mapping)

            result.overlaps = combine_result.overlaps

            # Report overlaps
            for overlap in combine_result.overlaps:
                print(
                    f"    WARNING: Overlap between '{overlap.mask1_name}' and "
                    f"'{overlap.mask2_name}': {overlap.overlap_voxels:,} voxels"
                )

            # Step 5: Convert to SimpleITK image with proper metadata
            label_image = sitk.GetImageFromArray(combine_result.combined_labels)
            label_image.CopyInformation(image)  # Copy spacing, origin, direction

            # Step 6: Validation
            validation_result = self.validator.validate_all(
                image, label_image, processed_masks
            )
            result.validation_result = validation_result

            # Check for errors
            if validation_result.errors:
                print(f"    Validation errors:")
                for issue in validation_result.errors:
                    print(f"      - {issue.message}")

                if self.config.strict_validation:
                    result.error_message = (
                        f"Validation failed: {len(validation_result.errors)} errors"
                    )
                    return result

            # Step 7: Write output files
            case_id = self.formatter.format_case_id(patient_dir.name)

            result.image_path = self.formatter.write_image(
                image,
                self.config.output_dir,
                case_id,
                compress=self.config.compress,
            )

            result.label_path = self.formatter.write_label(
                label_image,
                self.config.output_dir,
                case_id,
                compress=self.config.compress,
            )

            result.success = True
            print(f"    Output: {result.image_path.name}, {result.label_path.name}")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            print(f"    ERROR: {e}")

        return result

    def convert_dataset(self, batch_dir: Optional[Path] = None) -> list[ConversionResult]:
        """
        Convert all patients in a directory.

        Args:
            batch_dir: Directory containing patient folders. If None, uses config.input_dir

        Returns:
            List of ConversionResult for each patient
        """
        if batch_dir is None:
            batch_dir = self.config.input_dir

        batch_dir = Path(batch_dir)
        results = []

        # Find patient directories (directories that contain the image folder)
        patient_dirs = []
        for d in batch_dir.iterdir():
            if d.is_dir():
                image_folder = d / self.config.image_folder_name
                if image_folder.exists():
                    patient_dirs.append(d)

        if not patient_dirs:
            print(f"No patient directories found with image folder '{self.config.image_folder_name}'")
            return results

        print(f"Found {len(patient_dirs)} patient(s) to convert")
        print(f"Output directory: {self.config.output_dir}")
        print()

        for patient_dir in tqdm(patient_dirs, desc="Converting patients"):
            result = self.convert_patient(patient_dir)
            results.append(result)
            print()

        # Generate dataset.json
        self.formatter.write_dataset_json_from_results(
            self.config.output_dir,
            results,
            self.config.modality,
        )

        # Summary
        successful = sum(1 for r in results if r.success)
        print(f"\nConversion complete: {successful}/{len(results)} successful")

        if successful < len(results):
            print("\nFailed conversions:")
            for r in results:
                if not r.success:
                    print(f"  - {r.patient_id}: {r.error_message}")

        return results

    def _identify_folders(
        self, patient_dir: Path
    ) -> tuple[Optional[Path], list[Path]]:
        """
        Identify the image folder and mask folders.

        Args:
            patient_dir: Path to patient directory

        Returns:
            Tuple of (image_folder, mask_folders)
        """
        image_folder = patient_dir / self.config.image_folder_name

        if not image_folder.exists():
            return None, []

        # All other directories are mask folders
        mask_folders = []
        for d in patient_dir.iterdir():
            if d.is_dir() and d.name != self.config.image_folder_name:
                # Check if it looks like a mask folder (has numeric prefix)
                if d.name[0].isdigit():
                    mask_folders.append(d)

        # Sort by numeric prefix
        mask_folders.sort(key=lambda x: x.name)

        return image_folder, mask_folders
