"""
Label combination utilities.

This module combines multiple binary masks into a single multi-label
segmentation volume suitable for nnU-Net training.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class OverlapInfo:
    """Information about overlapping regions between masks."""

    mask1_name: str
    mask2_name: str
    overlap_voxels: int
    overlap_positions: Optional[np.ndarray] = None


@dataclass
class CombineResult:
    """Result of combining multiple masks."""

    combined_labels: np.ndarray
    label_mapping: dict[str, int]
    overlaps: list[OverlapInfo] = field(default_factory=list)
    unlabeled_voxels: int = 0
    total_voxels: int = 0


class LabelCombiner:
    """
    Combine multiple binary masks into a single multi-label segmentation.

    Handles folder naming convention where labels are prefixed with numbers
    (e.g., "01_femur", "02_tibia").
    """

    def __init__(self, background_label: int = 0):
        """
        Initialize the label combiner.

        Args:
            background_label: Value to use for unlabeled voxels (default 0)
        """
        self.background_label = background_label

    @staticmethod
    def parse_label_from_folder_name(folder_name: str) -> tuple[int, str]:
        """
        Parse label integer and name from folder name.

        Expected format: "XX_name" where XX is a number (e.g., "01_femur")

        Args:
            folder_name: Name of the mask folder

        Returns:
            Tuple of (label_integer, label_name)

        Raises:
            ValueError: If folder name doesn't match expected pattern
        """
        # Match pattern: one or more digits, underscore, then rest of name
        match = re.match(r"^(\d+)_(.+)$", folder_name)

        if match:
            label_int = int(match.group(1))
            label_name = match.group(2)
            return label_int, label_name

        # Try just extracting leading numbers
        match = re.match(r"^(\d+)", folder_name)
        if match:
            label_int = int(match.group(1))
            return label_int, folder_name

        raise ValueError(
            f"Folder name '{folder_name}' does not match expected pattern 'XX_name' "
            f"(e.g., '01_femur')"
        )

    def extract_label_mapping(self, folder_names: list[str]) -> dict[str, int]:
        """
        Extract label mapping from folder names.

        Args:
            folder_names: List of mask folder names

        Returns:
            Dictionary mapping folder name to label integer
        """
        mapping = {}

        for folder_name in folder_names:
            try:
                label_int, _ = self.parse_label_from_folder_name(folder_name)
                mapping[folder_name] = label_int
            except ValueError as e:
                print(f"Warning: {e}")
                continue

        return mapping

    def combine(
        self,
        masks: dict[str, np.ndarray],
        label_mapping: Optional[dict[str, int]] = None,
    ) -> CombineResult:
        """
        Combine multiple binary masks into a single multi-label segmentation.

        Args:
            masks: Dictionary mapping mask name to binary mask array
            label_mapping: Optional dictionary mapping mask name to label integer.
                          If None, will try to extract from mask names.

        Returns:
            CombineResult with combined labels and overlap information
        """
        if not masks:
            raise ValueError("No masks provided")

        # Get reference shape from first mask
        reference_shape = next(iter(masks.values())).shape

        # Verify all masks have the same shape
        for name, mask in masks.items():
            if mask.shape != reference_shape:
                raise ValueError(
                    f"Mask '{name}' has shape {mask.shape}, "
                    f"expected {reference_shape}"
                )

        # Extract label mapping if not provided
        if label_mapping is None:
            label_mapping = self.extract_label_mapping(list(masks.keys()))

        # Initialize output with background
        combined = np.full(reference_shape, self.background_label, dtype=np.uint8)

        # Track which voxels have been labeled (for overlap detection)
        labeled_tracker = np.zeros(reference_shape, dtype=bool)
        label_source = np.full(reference_shape, "", dtype=object)  # Track which mask labeled each voxel

        overlaps = []

        # Sort by label value for consistent processing order
        sorted_items = sorted(
            [(name, label_mapping.get(name, -1)) for name in masks.keys()],
            key=lambda x: x[1],
        )

        for mask_name, label_value in sorted_items:
            if label_value < 0:
                print(f"Warning: Skipping mask '{mask_name}' - no valid label mapping")
                continue

            if mask_name not in masks:
                continue

            mask = masks[mask_name].astype(bool)

            # Check for overlaps with previously labeled regions
            overlap_mask = labeled_tracker & mask
            overlap_count = np.sum(overlap_mask)

            if overlap_count > 0:
                # Find which masks overlap
                overlapping_sources = np.unique(label_source[overlap_mask])
                overlapping_sources = [s for s in overlapping_sources if s != ""]

                for other_name in overlapping_sources:
                    overlaps.append(
                        OverlapInfo(
                            mask1_name=other_name,
                            mask2_name=mask_name,
                            overlap_voxels=int(
                                np.sum((label_source == other_name) & mask)
                            ),
                        )
                    )

            # Apply label (later labels overwrite earlier ones in overlap regions)
            combined[mask] = label_value
            label_source[mask] = mask_name
            labeled_tracker |= mask

        # Count unlabeled voxels
        unlabeled_count = np.sum(~labeled_tracker)

        return CombineResult(
            combined_labels=combined,
            label_mapping=label_mapping,
            overlaps=overlaps,
            unlabeled_voxels=int(unlabeled_count),
            total_voxels=int(np.prod(reference_shape)),
        )

    def combine_from_folders(
        self,
        masks: dict[Path, np.ndarray],
        label_mapping: Optional[dict[str, int]] = None,
    ) -> CombineResult:
        """
        Combine masks using Path objects for folder names.

        Args:
            masks: Dictionary mapping Path to binary mask array
            label_mapping: Optional dictionary mapping folder name to label integer

        Returns:
            CombineResult with combined labels and overlap information
        """
        # Convert Path keys to string folder names
        masks_by_name = {path.name: mask for path, mask in masks.items()}

        return self.combine(masks_by_name, label_mapping)
