"""
Mimics mask processing utilities.

This module handles the extraction of binary masks from DICOM files
exported from Mimics software. Mimics exports masks where:
- The ROI has a uniform HU value
- Background/non-ROI areas have random noise values

The processor detects and extracts the uniform HU region.
"""

from collections import Counter
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


class MimicsMaskProcessor:
    """
    Process masks exported from Mimics software.

    Mimics exports masks as DICOM where the ROI has a uniform HU value
    and the background has random noise. This processor detects and
    extracts the uniform HU region.
    """

    def __init__(
        self,
        hu_tolerance: float = 0.5,
        min_region_size: int = 100,
        use_connected_components: bool = True,
    ):
        """
        Initialize the mask processor.

        Args:
            hu_tolerance: Tolerance for matching uniform HU values
            min_region_size: Minimum number of voxels for a valid region
            use_connected_components: Whether to filter by connected components
        """
        self.hu_tolerance = hu_tolerance
        self.min_region_size = min_region_size
        self.use_connected_components = use_connected_components

    def extract_mask(self, volume: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Extract the binary mask from a Mimics-exported volume.

        Args:
            volume: 3D numpy array from DICOM

        Returns:
            Tuple of (binary_mask, detected_hu) where:
            - binary_mask: Boolean array where True indicates ROI
            - detected_hu: The uniform HU value detected
        """
        # Step 1: Analyze the histogram to find the uniform HU value
        detected_hu = self._detect_uniform_hu(volume)

        # Step 2: Create initial mask based on detected HU
        mask = np.abs(volume - detected_hu) <= self.hu_tolerance

        # Step 3: Use connected component analysis to clean the mask
        if self.use_connected_components and np.sum(mask) > 0:
            mask = self._extract_largest_component(mask)

        # Step 4: Morphological cleaning (fill small holes, remove small objects)
        if np.sum(mask) > 0:
            mask = self._morphological_cleanup(mask)

        return mask.astype(bool), detected_hu

    def _detect_uniform_hu(self, volume: np.ndarray) -> float:
        """
        Detect the uniform HU value using histogram analysis.

        The strategy is to find the value that appears most frequently
        (excluding typical background values) and has low local variance,
        indicating a uniform region.

        Args:
            volume: 3D numpy array

        Returns:
            Detected uniform HU value
        """
        flat = volume.flatten()

        # Method 1: Simple mode detection (most frequent value)
        value_counts = Counter(flat)
        most_common = value_counts.most_common(20)

        if not most_common:
            return float(np.median(flat))

        # Filter out likely background values
        # Background noise typically has varied values, each appearing few times
        # The uniform region has one value appearing many times
        max_count = most_common[0][1]

        # Significant values should appear at least 5% as often as the most common
        significant_values = [(v, c) for v, c in most_common if c > max_count * 0.05]

        if not significant_values:
            return float(most_common[0][0])

        # Method 2: Local variance analysis to confirm
        # The true mask value will have lowest variance in its neighborhood
        best_value = None
        best_score = float("inf")

        for value, count in significant_values:
            # Check local variance of regions with this value
            candidate_mask = np.abs(volume - value) <= self.hu_tolerance

            if np.sum(candidate_mask) < self.min_region_size:
                continue

            # Calculate variance of values in the masked region
            region_values = volume[candidate_mask]
            variance = np.var(region_values)

            # Score: lower variance and higher count is better
            # We want the uniform region (low variance) with many voxels (high count)
            # Add small epsilon to avoid division by zero
            score = variance / (np.log1p(count) + 1e-10)

            if score < best_score:
                best_score = score
                best_value = value

        if best_value is None:
            best_value = most_common[0][0]

        return float(best_value)

    def _detect_uniform_hu_histogram(self, volume: np.ndarray) -> float:
        """
        Alternative method using histogram peak detection.

        More robust for cases where noise has uniform distribution.

        Args:
            volume: 3D numpy array

        Returns:
            Detected uniform HU value
        """
        flat = volume.flatten()

        # Create histogram with small bins
        hist, bin_edges = np.histogram(flat, bins=500)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth histogram to reduce noise effects
        from scipy.ndimage import gaussian_filter1d

        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)

        # Find peaks
        peaks, properties = find_peaks(
            hist_smooth, height=np.max(hist_smooth) * 0.1, distance=5
        )

        if len(peaks) == 0:
            return float(np.median(flat))

        # Return the highest peak (most voxels)
        peak_heights = hist_smooth[peaks]
        best_peak_idx = peaks[np.argmax(peak_heights)]

        return float(bin_centers[best_peak_idx])

    def _extract_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected component.

        Args:
            mask: Binary mask array

        Returns:
            Mask with only the largest connected component
        """
        # Use 6-connectivity for 3D (face-connected)
        structure = ndimage.generate_binary_structure(3, 1)
        labeled_array, num_features = ndimage.label(mask, structure=structure)

        if num_features == 0:
            return mask

        # Find the largest component
        component_sizes = ndimage.sum(mask, labeled_array, range(1, num_features + 1))
        largest_label = np.argmax(component_sizes) + 1

        return (labeled_array == largest_label).astype(mask.dtype)

    def _morphological_cleanup(
        self,
        mask: np.ndarray,
        fill_holes: bool = True,
        remove_small: bool = True,
    ) -> np.ndarray:
        """
        Apply morphological operations to clean the mask.

        Args:
            mask: Binary mask array
            fill_holes: Whether to fill holes in the mask
            remove_small: Whether to remove small disconnected objects

        Returns:
            Cleaned mask
        """
        result = mask.copy()

        if fill_holes:
            # Fill holes slice by slice (faster than 3D and often sufficient)
            for i in range(result.shape[0]):
                result[i] = ndimage.binary_fill_holes(result[i])

        if remove_small:
            # Remove small objects
            structure = ndimage.generate_binary_structure(3, 1)
            labeled, num_features = ndimage.label(result, structure=structure)

            if num_features > 1:
                sizes = ndimage.sum(result, labeled, range(1, num_features + 1))
                # Keep only components larger than min_region_size
                for label_id in range(1, num_features + 1):
                    if sizes[label_id - 1] < self.min_region_size:
                        result[labeled == label_id] = False

        return result.astype(bool)

    def analyze_mask_statistics(self, volume: np.ndarray) -> dict:
        """
        Analyze the volume to provide statistics about potential mask regions.

        Useful for debugging and understanding the data.

        Args:
            volume: 3D numpy array

        Returns:
            Dictionary with analysis results
        """
        flat = volume.flatten()
        value_counts = Counter(flat)
        most_common = value_counts.most_common(10)

        # Extract mask
        mask, detected_hu = self.extract_mask(volume)

        stats = {
            "volume_shape": volume.shape,
            "total_voxels": volume.size,
            "unique_values": len(value_counts),
            "most_common_values": most_common,
            "detected_hu": detected_hu,
            "mask_voxels": int(np.sum(mask)),
            "mask_percentage": float(np.sum(mask) / volume.size * 100),
            "min_value": float(np.min(volume)),
            "max_value": float(np.max(volume)),
            "mean_value": float(np.mean(volume)),
        }

        return stats
