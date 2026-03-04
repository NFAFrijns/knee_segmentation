"""
Knee Segmentation: DICOM to MHA conversion tool for nnU-Net training.

This package provides tools to convert DICOM files exported from Mimics
to MHA format compatible with nnU-Net training.
"""

from .converter import DicomToMhaConverter, ConversionConfig, ConversionResult
from .dicom_reader import DicomSeriesReader
from .mask_processor import MimicsMaskProcessor
from .label_combiner import LabelCombiner
from .validator import SegmentationValidator, ValidationIssue, ValidationSeverity

__version__ = "0.1.0"

__all__ = [
    "DicomToMhaConverter",
    "ConversionConfig",
    "ConversionResult",
    "DicomSeriesReader",
    "MimicsMaskProcessor",
    "LabelCombiner",
    "SegmentationValidator",
    "ValidationIssue",
    "ValidationSeverity",
]
