"""
Validation utilities for segmentation data.

This module provides comprehensive validation checks to ensure
the converted data is compatible with nnU-Net training.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import SimpleITK as sitk


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found during checking."""

    severity: ValidationSeverity
    message: str
    location: str = ""
    suggestion: str = ""

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        msg = f"{prefix} {self.message}"
        if self.location:
            msg += f" (at {self.location})"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


@dataclass
class ValidationResult:
    """Result of validation checks."""

    issues: list[ValidationIssue] = field(default_factory=list)
    passed: bool = True

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get all error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get all warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def infos(self) -> list[ValidationIssue]:
        """Get all info-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.passed = False


class SegmentationValidator:
    """Comprehensive validation for nnU-Net compatibility."""

    def __init__(self, strict: bool = True):
        """
        Initialize the validator.

        Args:
            strict: If True, overlaps are errors. If False, overlaps are warnings.
        """
        self.strict = strict

    def validate_all(
        self,
        image: sitk.Image,
        label: sitk.Image,
        masks: Optional[dict[str, np.ndarray]] = None,
    ) -> ValidationResult:
        """
        Run all validation checks.

        Args:
            image: The main image (SimpleITK Image)
            label: The label/segmentation image (SimpleITK Image)
            masks: Optional dictionary of individual binary masks

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult()

        self._check_dimensions_match(image, label, result)
        self._check_spacing_match(image, label, result)
        self._check_origin_direction_match(image, label, result)
        self._check_label_data_type(label, result)

        # Get label array for content checks
        label_array = sitk.GetArrayFromImage(label)
        self._check_label_values(label_array, result)

        if masks:
            self._check_complete_coverage(masks, label_array.shape, result)
            self._check_overlapping_labels(masks, result)

        return result

    def _check_dimensions_match(
        self,
        image: sitk.Image,
        label: sitk.Image,
        result: ValidationResult,
    ):
        """Verify image and label have identical dimensions."""
        if image.GetSize() != label.GetSize():
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Dimension mismatch: image {image.GetSize()} vs label {label.GetSize()}",
                    suggestion="Check that all DICOM series have the same number of slices",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Dimensions match: {image.GetSize()}",
                )
            )

    def _check_spacing_match(
        self,
        image: sitk.Image,
        label: sitk.Image,
        result: ValidationResult,
    ):
        """Verify image and label have matching spacing."""
        image_spacing = image.GetSpacing()
        label_spacing = label.GetSpacing()

        # Allow small tolerance for floating point differences
        tolerance = 1e-5
        spacing_match = all(
            abs(a - b) < tolerance for a, b in zip(image_spacing, label_spacing)
        )

        if not spacing_match:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Spacing mismatch: image {image_spacing} vs label {label_spacing}",
                    suggestion="Ensure all DICOM series have the same voxel spacing",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Spacing match: {image_spacing}",
                )
            )

    def _check_origin_direction_match(
        self,
        image: sitk.Image,
        label: sitk.Image,
        result: ValidationResult,
    ):
        """Verify image and label have matching origin and direction."""
        image_origin = image.GetOrigin()
        label_origin = label.GetOrigin()

        tolerance = 1e-3
        origin_match = all(
            abs(a - b) < tolerance for a, b in zip(image_origin, label_origin)
        )

        if not origin_match:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Origin mismatch: image {image_origin} vs label {label_origin}",
                    suggestion="This may cause misalignment in visualization",
                )
            )

        image_direction = image.GetDirection()
        label_direction = label.GetDirection()

        direction_match = all(
            abs(a - b) < tolerance for a, b in zip(image_direction, label_direction)
        )

        if not direction_match:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Direction matrix mismatch between image and label",
                    suggestion="This may cause misalignment in visualization",
                )
            )

    def _check_label_data_type(self, label: sitk.Image, result: ValidationResult):
        """Verify label has appropriate integer data type."""
        pixel_type = label.GetPixelIDTypeAsString()

        if "float" in pixel_type.lower():
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Label has float type ({pixel_type}), should be integer",
                    suggestion="Labels must be integer type for nnU-Net",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Label data type: {pixel_type}",
                )
            )

    def _check_label_values(self, label_array: np.ndarray, result: ValidationResult):
        """Check that label values are valid."""
        unique_labels = np.unique(label_array)

        result.add_issue(
            ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Unique label values: {unique_labels.tolist()}",
            )
        )

        # Check for negative labels
        if np.any(unique_labels < 0):
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Negative label values found: {unique_labels[unique_labels < 0].tolist()}",
                    suggestion="All label values should be non-negative integers",
                )
            )

        # Note: We don't require consecutive labels as per user requirement

    def _check_complete_coverage(
        self,
        masks: dict[str, np.ndarray],
        expected_shape: tuple,
        result: ValidationResult,
    ):
        """Check if all voxels are covered by exactly one label."""
        # Combine all masks
        combined = np.zeros(expected_shape, dtype=bool)
        for mask in masks.values():
            combined |= mask.astype(bool)

        unlabeled_count = np.sum(~combined)
        total_voxels = np.prod(expected_shape)
        coverage_pct = ((total_voxels - unlabeled_count) / total_voxels) * 100

        if unlabeled_count > 0:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"{unlabeled_count:,} voxels ({100-coverage_pct:.2f}%) have no label",
                    suggestion="Ensure background mask (00_*) covers all non-ROI voxels",
                )
            )
        else:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Complete coverage: 100% of voxels are labeled",
                )
            )

    def _check_overlapping_labels(
        self,
        masks: dict[str, np.ndarray],
        result: ValidationResult,
    ):
        """Check for overlapping regions between masks."""
        mask_names = list(masks.keys())
        overlap_found = False

        for i, name1 in enumerate(mask_names):
            for name2 in mask_names[i + 1 :]:
                mask1 = masks[name1].astype(bool)
                mask2 = masks[name2].astype(bool)

                overlap = mask1 & mask2
                overlap_count = np.sum(overlap)

                if overlap_count > 0:
                    overlap_found = True
                    severity = (
                        ValidationSeverity.ERROR
                        if self.strict
                        else ValidationSeverity.WARNING
                    )
                    result.add_issue(
                        ValidationIssue(
                            severity=severity,
                            message=f"Overlap between '{name1}' and '{name2}': {overlap_count:,} voxels",
                            suggestion="Review segmentation in Mimics to resolve overlaps",
                        )
                    )

        if not overlap_found:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="No overlapping regions between masks",
                )
            )


def validate_nnunet_dataset(dataset_dir) -> ValidationResult:
    """
    Validate an nnU-Net dataset directory structure.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        ValidationResult with all issues found
    """
    from pathlib import Path

    dataset_dir = Path(dataset_dir)
    result = ValidationResult()

    # Check required structure
    required_items = ["imagesTr", "labelsTr", "dataset.json"]
    for item in required_items:
        path = dataset_dir / item
        if not path.exists():
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required item: {item}",
                    location=str(dataset_dir),
                )
            )

    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    if images_dir.exists() and labels_dir.exists():
        # Get case IDs from images (remove _0000 suffix)
        images = {f.stem.replace("_0000", "") for f in images_dir.glob("*.mha")}
        labels = {f.stem for f in labels_dir.glob("*.mha")}

        # Check for unpaired files
        missing_labels = images - labels
        missing_images = labels - images

        for case_id in missing_labels:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Image without corresponding label: {case_id}",
                )
            )

        for case_id in missing_images:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Label without corresponding image: {case_id}",
                )
            )

        if not missing_labels and not missing_images:
            result.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"All {len(images)} cases have matching image-label pairs",
                )
            )

    return result
