# Knee Segmentation - DICOM to MHA Converter

A Python tool to convert DICOM files exported from **Mimics** to MHA format compatible with **nnU-Net** training.

## Features

- Converts DICOM series to MHA format
- Handles Mimics mask exports (detects uniform HU regions, filters out noise)
- Combines multiple binary masks into multi-label segmentation
- Extracts label values from folder names (e.g., `01_femur` → label 1)
- Validates output for nnU-Net compatibility
- Generates `dataset.json` automatically

## Installation

```bash
# Clone the repository
git clone git@github.com-personal:NFAFrijns/knee_segmentation.git
cd knee_segmentation

# Install in development mode
pip install -e .
```

## Input Data Structure

Organize your DICOM data as follows:

```
input_data/
├── patient_001/
│   ├── CT_scan/              # Main DICOM image (name specified via --image-folder)
│   │   ├── IM_0001.dcm
│   │   ├── IM_0002.dcm
│   │   └── ...
│   ├── 00_background/        # Background mask (label 0)
│   │   └── *.dcm
│   ├── 01_femur/             # Femur mask (label 1)
│   │   └── *.dcm
│   ├── 02_tibia/             # Tibia mask (label 2)
│   │   └── *.dcm
│   └── 05_cartilage/         # Cartilage mask (label 5) - gaps allowed
│       └── *.dcm
├── patient_002/
│   └── ...
```

**Notes:**
- Mask folder names must start with a number prefix (e.g., `01_`, `02_`)
- The number determines the label value in the output segmentation
- Not all labels need to be present in every patient (gaps allowed)
- Background should be provided as `00_background`

## Output Structure (nnU-Net format)

```
output/
├── dataset.json
├── imagesTr/
│   ├── case001_0000.mha
│   ├── case002_0000.mha
│   └── ...
└── labelsTr/
    ├── case001.mha
    ├── case002.mha
    └── ...
```

## Usage

### Convert Dataset

```bash
dicom2mha convert ./input_data ./output --image-folder "CT_scan"
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--image-folder`, `-i` | Name of the DICOM image folder (required) | - |
| `--dataset-name`, `-n` | Name for the nnU-Net dataset | KneeSegmentation |
| `--dataset-id`, `-d` | Dataset ID (001-999) | 1 |
| `--modality`, `-m` | Imaging modality (CT or MR) | CT |
| `--hu-tolerance`, `-t` | Tolerance for uniform HU detection | 0.5 |
| `--strict/--no-strict` | Fail on validation errors | --no-strict |

### Inspect MHA File

View properties of an MHA file:

```bash
dicom2mha inspect ./output/imagesTr/case001_0000.mha
```

### Validate Dataset

Check if the output is valid for nnU-Net:

```bash
dicom2mha validate ./output
```

### Analyze Mask (Debugging)

Understand the HU distribution in a mask folder:

```bash
dicom2mha analyze-mask ./input_data/patient_001/01_femur
```

## How It Works

### Mimics Mask Processing

Mimics exports masks as DICOM files where:
- The region of interest (ROI) has a **uniform HU value**
- Background areas have **random noise values**

The tool:
1. Analyzes the histogram to find the most frequent value
2. Creates a binary mask for voxels matching that value
3. Uses connected component analysis to keep the largest region
4. Applies morphological cleanup (fill holes, remove small objects)

### Validation Checks

1. **Dimension match**: Image and all masks have same dimensions
2. **Spacing match**: All files have same voxel spacing
3. **Overlap detection**: Warns if masks overlap (reports which masks)
4. **Complete coverage**: Checks that all voxels have exactly one label

## Dependencies

- Python >= 3.10
- numpy >= 1.24.0
- SimpleITK >= 2.3.0
- pydicom >= 2.4.0
- scipy >= 1.10.0
- click >= 8.1.0
- rich >= 13.0.0
- tqdm >= 4.65.0

## Python API

```python
from pathlib import Path
from knee_segmentation import DicomToMhaConverter, ConversionConfig

config = ConversionConfig(
    input_dir=Path("./input_data"),
    output_dir=Path("./output"),
    image_folder_name="CT_scan",
    dataset_name="KneeSegmentation",
    modality="CT",
)

converter = DicomToMhaConverter(config)
results = converter.convert_dataset()

for result in results:
    if result.success:
        print(f"{result.patient_id}: OK")
    else:
        print(f"{result.patient_id}: FAILED - {result.error_message}")
```

## License

MIT
