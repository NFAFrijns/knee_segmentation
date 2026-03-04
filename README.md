# Knee Segmentation - DICOM to MHA Converter

A Python tool to convert DICOM files exported from **Mimics** to MHA format compatible with **nnU-Net** training.

## Features

- Converts DICOM series to MHA format
- Handles Mimics mask exports (detects uniform HU regions, filters out noise)
- Combines multiple binary masks into multi-label segmentation
- Extracts label values from folder names (e.g., `01_femur` → label 1)
- Validates output for nnU-Net compatibility
- Generates `dataset.json` automatically

---

## Prerequisites

- **Python 3.10 or higher**
- **pip** (Python package installer)
- **Git** (for cloning the repository)

Check your Python version:
```bash
python --version
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/NFAFrijns/knee_segmentation.git
```

### Step 2: Navigate to the Project Directory

```bash
cd knee_segmentation
```

### Step 3: (Recommended) Create a Virtual Environment

**Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 4: Install the Package

```bash
pip install -e .
```

This will install all required dependencies automatically.

### Step 5: Verify Installation

```bash
dicom2mha --help
```

You should see:
```
Usage: dicom2mha [OPTIONS] COMMAND [ARGS]...

  DICOM to MHA conversion tool for nnU-Net training.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  analyze-mask  Analyze a DICOM mask directory to understand its structure.
  convert       Convert DICOM files to MHA format for nnU-Net.
  inspect       Inspect an MHA file and display its properties.
  validate      Validate an nnU-Net dataset structure.
```

---

## Preparing Your Data

### Required Input Structure

Your DICOM data must be organized in a specific folder structure:

```
input_data/                          # Your input directory
│
├── patient_001/                     # First patient folder (name can be anything)
│   │
│   ├── CT_scan/                     # Main DICOM image folder
│   │   ├── IM_0001.dcm              #   (name specified via --image-folder)
│   │   ├── IM_0002.dcm
│   │   ├── IM_0003.dcm
│   │   └── ...
│   │
│   ├── 00_background/               # Background mask - label 0
│   │   ├── IM_0001.dcm
│   │   ├── IM_0002.dcm
│   │   └── ...
│   │
│   ├── 01_femur/                    # Femur mask - label 1
│   │   ├── IM_0001.dcm
│   │   └── ...
│   │
│   ├── 02_tibia/                    # Tibia mask - label 2
│   │   ├── IM_0001.dcm
│   │   └── ...
│   │
│   └── 03_patella/                  # Patella mask - label 3
│       ├── IM_0001.dcm
│       └── ...
│
├── patient_002/                     # Second patient folder
│   ├── CT_scan/
│   ├── 00_background/
│   ├── 01_femur/
│   └── ...                          # Not all masks required for each patient
│
└── patient_003/
    └── ...
```

### Important Rules

1. **Image folder**: The main DICOM series folder name must be the same for all patients (e.g., `CT_scan`)

2. **Mask folder naming**:
   - Must start with a number: `00_`, `01_`, `02_`, etc.
   - The number becomes the label value in the output
   - Format: `XX_name` where XX is the label number

3. **Background mask**: Should be named `00_background` (label 0)

4. **Gaps allowed**: Not every patient needs all masks (e.g., patient_001 has labels 0,1,2,3 but patient_002 only has 0,1,2)

5. **All masks must have same dimensions**: Each mask DICOM series must have the same number of slices as the main image

---

## Usage

### Basic Conversion

Convert all patients in a directory:

```bash
dicom2mha convert INPUT_DIR OUTPUT_DIR --image-folder "FOLDER_NAME"
```

**Example:**
```bash
dicom2mha convert ./input_data ./output --image-folder "CT_scan"
```

### Full Example with All Options

```bash
dicom2mha convert ./input_data ./output --image-folder "CT_scan" --dataset-name "KneeSegmentation" --dataset-id 1 --modality CT --hu-tolerance 0.5 --min-mask-size 100 --no-strict
```

### Command Options

| Option | Short | Description | Default | Required |
|--------|-------|-------------|---------|----------|
| `--image-folder` | `-i` | Name of the DICOM image folder | - | **Yes** |
| `--dataset-name` | `-n` | Name for the nnU-Net dataset | KneeSegmentation | No |
| `--dataset-id` | `-d` | Dataset ID number (1-999) | 1 | No |
| `--modality` | `-m` | Imaging modality: CT or MR | CT | No |
| `--hu-tolerance` | `-t` | Tolerance for uniform HU detection | 0.5 | No |
| `--min-mask-size` | | Minimum voxels for valid mask | 100 | No |
| `--strict` | | Fail on any validation error | False | No |
| `--no-compress` | | Disable MHA compression | False | No |

---

## Output Structure

After conversion, you'll have an nnU-Net-compatible dataset:

```
output/
│
├── dataset.json                     # Dataset configuration for nnU-Net
│
├── imagesTr/                        # Training images
│   ├── case001_0000.mha             # Patient 1 image
│   ├── case002_0000.mha             # Patient 2 image
│   └── ...
│
└── labelsTr/                        # Training labels (segmentations)
    ├── case001.mha                  # Patient 1 combined labels
    ├── case002.mha                  # Patient 2 combined labels
    └── ...
```

### dataset.json Example

```json
{
  "channel_names": {
    "0": "CT"
  },
  "labels": {
    "background": 0,
    "femur": 1,
    "tibia": 2,
    "patella": 3
  },
  "numTraining": 3,
  "file_ending": ".mha"
}
```

---

## Other Commands

### Inspect an MHA File

View detailed properties of any MHA file:

```bash
dicom2mha inspect ./output/imagesTr/case001_0000.mha
```

Output:
```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property           ┃ Value                       ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Size (x, y, z)     │ (512, 512, 100)             │
│ Spacing (mm)       │ (0.5, 0.5, 1.0)             │
│ Pixel Type         │ 16-bit signed integer       │
│ Min Value          │ -1024.00                    │
│ Max Value          │ 2048.00                     │
│ Unique Values      │ 4                           │
│ Labels             │ [0, 1, 2, 3]                │
└────────────────────┴─────────────────────────────┘
```

### Validate Dataset

Check if output is valid for nnU-Net:

```bash
dicom2mha validate ./output
```

### Analyze a Mask (Debugging)

If mask extraction isn't working correctly, analyze the raw DICOM:

```bash
dicom2mha analyze-mask ./input_data/patient_001/01_femur
```

This shows the HU distribution and detected uniform value.

---

## How It Works

### Mimics Mask Processing

When Mimics exports masks as DICOM:
- The **region of interest (ROI)** has a **uniform HU value** (e.g., all voxels = 100)
- **Background areas** have **random noise values** (varying HU values)

The tool automatically:
1. Analyzes the histogram to find the most frequent (uniform) value
2. Creates a binary mask for voxels matching that value
3. Uses connected component analysis to keep only the largest region
4. Fills small holes and removes small disconnected objects

### Validation Checks

The tool validates:
1. **Dimensions match**: Image and all masks have same size
2. **Spacing match**: All files have same voxel spacing
3. **Overlap detection**: Warns if any masks overlap
4. **Complete coverage**: Checks all voxels have exactly one label

---

## Python API

You can also use the tool programmatically:

```python
from pathlib import Path
from knee_segmentation import DicomToMhaConverter, ConversionConfig

# Configure the conversion
config = ConversionConfig(
    input_dir=Path("./input_data"),
    output_dir=Path("./output"),
    image_folder_name="CT_scan",      # Name of your image folder
    dataset_name="KneeSegmentation",
    dataset_id=1,
    modality="CT",
    hu_tolerance=0.5,
    min_mask_size=100,
    strict_validation=False,
    compress=True,
)

# Run conversion
converter = DicomToMhaConverter(config)
results = converter.convert_dataset()

# Check results
for result in results:
    if result.success:
        print(f"{result.patient_id}: SUCCESS")
        print(f"  Image: {result.image_path}")
        print(f"  Label: {result.label_path}")
    else:
        print(f"{result.patient_id}: FAILED - {result.error_message}")
```

---

## Troubleshooting

### "No patient directories found"

**Cause**: The tool can't find folders containing your image folder.

**Solution**: Make sure:
- Your input directory contains patient folders
- Each patient folder contains a subfolder with the exact name you specified in `--image-folder`

```bash
# Check your structure (Windows)
dir .\input_data\patient_001\

# Check your structure (macOS/Linux)
ls ./input_data/patient_001/

# Should show: CT_scan/  00_background/  01_femur/  etc.
```

### "No DICOM series found"

**Cause**: The folder doesn't contain valid DICOM files.

**Solution**:
- Check that files have `.dcm` or `.DCM` extension
- Verify files are valid DICOM (not corrupted)

### "Dimension mismatch"

**Cause**: A mask has different dimensions than the main image.

**Solution**: Re-export the mask from Mimics with the same slice count as the original image.

### "Command not found: dicom2mha"

**Cause**: Package not installed or virtual environment not activated.

**Solution**:
```bash
# Activate virtual environment first (Windows)
venv\Scripts\activate

# Activate virtual environment first (macOS/Linux)
source venv/bin/activate

# Then verify or reinstall
pip install -e .
```

### Mask extraction not working correctly

**Debug**: Use the analyze command to see what the tool detects:
```bash
dicom2mha analyze-mask ./input_data/patient_001/01_femur
```

Check if the "Detected Uniform HU" value makes sense for your mask.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24.0 | Array operations |
| SimpleITK | >= 2.3.0 | DICOM/MHA reading and writing |
| pydicom | >= 2.4.0 | DICOM metadata extraction |
| scipy | >= 1.10.0 | Image processing (morphology) |
| click | >= 8.1.0 | Command-line interface |
| rich | >= 13.0.0 | Terminal formatting |
| tqdm | >= 4.65.0 | Progress bars |

---

## License

MIT
