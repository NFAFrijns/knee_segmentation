"""
DICOM series reading utilities.

This module provides functionality to read DICOM series using SimpleITK,
with robust handling of various DICOM formats and configurations.
"""

from pathlib import Path
from typing import Optional

import pydicom
import SimpleITK as sitk


class DicomSeriesReader:
    """Read DICOM series with robust handling of various formats."""

    def __init__(self):
        """Initialize the DICOM series reader."""
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()

    def read_dicom_series(
        self,
        directory: Path,
        series_id: Optional[str] = None,
    ) -> sitk.Image:
        """
        Read a DICOM series from a directory.

        Uses SimpleITK's GDCM-based reader which handles:
        - Proper slice ordering
        - Multi-frame DICOM
        - Various transfer syntaxes

        Args:
            directory: Path to directory containing DICOM files
            series_id: Optional specific series ID to read. If None, uses the first series.

        Returns:
            SimpleITK Image object

        Raises:
            ValueError: If no DICOM series found in directory
        """
        directory = Path(directory)

        # Get all series IDs in the directory
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory))

        if not series_ids:
            raise ValueError(f"No DICOM series found in {directory}")

        if series_id is None:
            # Use the first (or only) series
            series_id = series_ids[0]
            if len(series_ids) > 1:
                print(f"Warning: Multiple series found in {directory}, using first: {series_id}")

        # Get file names sorted in the correct order
        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_id)

        self.reader.SetFileNames(dicom_names)

        # Execute the read
        image = self.reader.Execute()

        return image

    def get_series_ids(self, directory: Path) -> list[str]:
        """
        Get all series IDs in a directory.

        Args:
            directory: Path to directory containing DICOM files

        Returns:
            List of series ID strings
        """
        return list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)))

    def get_dicom_metadata(self, directory: Path) -> dict:
        """
        Extract relevant metadata from DICOM files.

        Args:
            directory: Path to directory containing DICOM files

        Returns:
            Dictionary with metadata fields
        """
        directory = Path(directory)
        dcm_files = list(directory.glob("*.dcm")) + list(directory.glob("*.DCM"))

        if not dcm_files:
            return {}

        # Read first file for metadata
        ds = pydicom.dcmread(str(dcm_files[0]))

        metadata = {
            "patient_id": getattr(ds, "PatientID", "Unknown"),
            "study_date": getattr(ds, "StudyDate", "Unknown"),
            "modality": getattr(ds, "Modality", "Unknown"),
            "manufacturer": getattr(ds, "Manufacturer", "Unknown"),
            "slice_thickness": getattr(ds, "SliceThickness", None),
            "pixel_spacing": list(ds.PixelSpacing) if hasattr(ds, "PixelSpacing") else None,
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
            "bits_allocated": getattr(ds, "BitsAllocated", None),
            "bits_stored": getattr(ds, "BitsStored", None),
        }

        return metadata

    def count_dicom_files(self, directory: Path) -> int:
        """
        Count DICOM files in a directory.

        Args:
            directory: Path to directory

        Returns:
            Number of DICOM files found
        """
        directory = Path(directory)
        dcm_count = len(list(directory.glob("*.dcm"))) + len(list(directory.glob("*.DCM")))
        return dcm_count
