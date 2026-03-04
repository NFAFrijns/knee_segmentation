"""
nnU-Net formatting utilities.

This module handles the output formatting to ensure compatibility
with nnU-Net training requirements.
"""

import json
import re
from pathlib import Path
from typing import Optional

import SimpleITK as sitk


class NnunetFormatter:
    """Format output for nnU-Net compatibility."""

    def __init__(self, dataset_name: str, dataset_id: int = 1):
        """
        Initialize the formatter.

        Args:
            dataset_name: Name for the nnU-Net dataset (e.g., "KneeSegmentation")
            dataset_id: Dataset ID number (1-999)
        """
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id

    def get_dataset_folder_name(self) -> str:
        """
        Generate nnU-Net dataset folder name.

        Returns:
            Folder name in format "DatasetXXX_Name"
        """
        return f"Dataset{self.dataset_id:03d}_{self.dataset_name}"

    def format_case_id(self, original_id: str, padding: int = 3) -> str:
        """
        Format case ID for nnU-Net.

        Converts various ID formats to consistent naming.

        Args:
            original_id: Original patient/case identifier
            padding: Number of digits for case number (default 3)

        Returns:
            Formatted case ID (e.g., "case001")
        """
        # Extract numeric part
        numbers = re.findall(r"\d+", original_id)

        if numbers:
            # Use the last number found
            num = int(numbers[-1])
            return f"case{num:0{padding}d}"
        else:
            # No number found, use hash-based ID
            return f"case{abs(hash(original_id)) % 1000:03d}"

    def get_image_filename(self, case_id: str, channel: int = 0) -> str:
        """
        Generate image filename for nnU-Net.

        Args:
            case_id: Case identifier
            channel: Channel number (default 0 for single-channel)

        Returns:
            Filename in format "caseXXX_CCCC.mha"
        """
        return f"{case_id}_{channel:04d}.mha"

    def get_label_filename(self, case_id: str) -> str:
        """
        Generate label filename for nnU-Net.

        Args:
            case_id: Case identifier

        Returns:
            Filename in format "caseXXX.mha"
        """
        return f"{case_id}.mha"

    def create_output_directories(self, output_dir: Path) -> tuple[Path, Path]:
        """
        Create the required output directory structure.

        Args:
            output_dir: Base output directory

        Returns:
            Tuple of (imagesTr_path, labelsTr_path)
        """
        output_dir = Path(output_dir)

        images_dir = output_dir / "imagesTr"
        labels_dir = output_dir / "labelsTr"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        return images_dir, labels_dir

    def write_image(
        self,
        image: sitk.Image,
        output_dir: Path,
        case_id: str,
        channel: int = 0,
        compress: bool = True,
    ) -> Path:
        """
        Write image to nnU-Net format.

        Args:
            image: SimpleITK Image to write
            output_dir: Base output directory
            case_id: Case identifier
            channel: Channel number
            compress: Whether to use compression

        Returns:
            Path to the written file
        """
        images_dir = Path(output_dir) / "imagesTr"
        images_dir.mkdir(parents=True, exist_ok=True)

        filename = self.get_image_filename(case_id, channel)
        filepath = images_dir / filename

        sitk.WriteImage(image, str(filepath), useCompression=compress)

        return filepath

    def write_label(
        self,
        label: sitk.Image,
        output_dir: Path,
        case_id: str,
        compress: bool = True,
    ) -> Path:
        """
        Write label to nnU-Net format.

        Args:
            label: SimpleITK Image (segmentation) to write
            output_dir: Base output directory
            case_id: Case identifier
            compress: Whether to use compression

        Returns:
            Path to the written file
        """
        labels_dir = Path(output_dir) / "labelsTr"
        labels_dir.mkdir(parents=True, exist_ok=True)

        filename = self.get_label_filename(case_id)
        filepath = labels_dir / filename

        sitk.WriteImage(label, str(filepath), useCompression=compress)

        return filepath

    def write_dataset_json(
        self,
        output_dir: Path,
        label_mapping: dict[str, int],
        modality: str = "CT",
        num_training: Optional[int] = None,
    ) -> Path:
        """
        Generate and write dataset.json file for nnU-Net.

        Args:
            output_dir: Base output directory
            label_mapping: Dictionary mapping label names to integers
            modality: Imaging modality (e.g., "CT", "MR")
            num_training: Number of training cases. If None, counts files in imagesTr.

        Returns:
            Path to the written dataset.json file
        """
        output_dir = Path(output_dir)

        # Create labels dict with background
        labels = {"background": 0}

        # Add other labels, converting folder names to clean label names
        for folder_name, label_int in label_mapping.items():
            # Extract clean name from folder (e.g., "01_femur" -> "femur")
            match = re.match(r"^\d+_(.+)$", folder_name)
            if match:
                clean_name = match.group(1)
            else:
                clean_name = folder_name

            labels[clean_name] = label_int

        # Sort labels by value
        labels = dict(sorted(labels.items(), key=lambda x: x[1]))

        # Count training cases if not provided
        if num_training is None:
            images_dir = output_dir / "imagesTr"
            if images_dir.exists():
                num_training = len(list(images_dir.glob("*.mha")))
            else:
                num_training = 0

        # Build dataset.json structure
        dataset_json = {
            "channel_names": {"0": modality},
            "labels": labels,
            "numTraining": num_training,
            "file_ending": ".mha",
        }

        # Write JSON
        json_path = output_dir / "dataset.json"
        with open(json_path, "w") as f:
            json.dump(dataset_json, f, indent=2)

        return json_path

    def write_dataset_json_from_results(
        self,
        output_dir: Path,
        results: list,  # List of ConversionResult
        modality: str = "CT",
    ) -> Path:
        """
        Generate dataset.json from conversion results.

        Args:
            output_dir: Base output directory
            results: List of ConversionResult objects
            modality: Imaging modality

        Returns:
            Path to the written dataset.json file
        """
        # Collect all labels from successful conversions
        all_labels = {}

        for result in results:
            if result.success and result.label_mapping:
                all_labels.update(result.label_mapping)

        num_training = sum(1 for r in results if r.success)

        return self.write_dataset_json(
            output_dir=output_dir,
            label_mapping=all_labels,
            modality=modality,
            num_training=num_training,
        )
