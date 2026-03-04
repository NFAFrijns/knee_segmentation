"""
Command-line interface for DICOM to MHA conversion.

This module provides a CLI for converting DICOM files exported from
Mimics to MHA format compatible with nnU-Net training.
"""

from pathlib import Path

import click
import numpy as np
import SimpleITK as sitk
from rich.console import Console
from rich.table import Table

from .converter import ConversionConfig, DicomToMhaConverter
from .validator import validate_nnunet_dataset, ValidationSeverity

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """DICOM to MHA conversion tool for nnU-Net training.

    Convert DICOM files exported from Mimics to MHA format compatible
    with nnU-Net. Handles uniform HU detection in masks and validates
    output for nnU-Net compatibility.
    """
    pass


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--image-folder",
    "-i",
    required=True,
    help="Name of the folder containing the main DICOM image series",
)
@click.option(
    "--dataset-name",
    "-n",
    default="KneeSegmentation",
    help="Name for the nnU-Net dataset",
)
@click.option(
    "--dataset-id",
    "-d",
    default=1,
    type=int,
    help="Dataset ID (001-999)",
)
@click.option(
    "--modality",
    "-m",
    default="CT",
    type=click.Choice(["CT", "MR"]),
    help="Imaging modality",
)
@click.option(
    "--hu-tolerance",
    "-t",
    default=0.5,
    type=float,
    help="Tolerance for uniform HU detection in masks",
)
@click.option(
    "--min-mask-size",
    default=100,
    type=int,
    help="Minimum voxels for a valid mask region",
)
@click.option(
    "--strict/--no-strict",
    default=False,
    help="Fail on validation errors (overlaps, etc.)",
)
@click.option(
    "--no-compress",
    is_flag=True,
    help="Disable compression in output MHA files",
)
def convert(
    input_dir: Path,
    output_dir: Path,
    image_folder: str,
    dataset_name: str,
    dataset_id: int,
    modality: str,
    hu_tolerance: float,
    min_mask_size: int,
    strict: bool,
    no_compress: bool,
):
    """Convert DICOM files to MHA format for nnU-Net.

    INPUT_DIR: Directory containing patient folders with DICOM data

    OUTPUT_DIR: Output directory for nnU-Net dataset

    Expected input structure:

    \b
    INPUT_DIR/
    ├── patient_001/
    │   ├── <image-folder>/     # Main DICOM (specified by --image-folder)
    │   ├── 00_background/      # Background mask (label 0)
    │   ├── 01_femur/           # Mask with label 1
    │   └── 02_tibia/           # Mask with label 2
    └── patient_002/
        └── ...

    Mask folders must be named with numeric prefix (e.g., "01_femur").
    The number determines the label value.
    """
    console.print("[bold blue]DICOM to MHA Converter[/bold blue]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")
    console.print(f"Image folder: {image_folder}")
    console.print()

    # Create configuration
    config = ConversionConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        image_folder_name=image_folder,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        modality=modality,
        hu_tolerance=hu_tolerance,
        min_mask_size=min_mask_size,
        strict_validation=strict,
        compress=not no_compress,
    )

    # Run conversion
    converter = DicomToMhaConverter(config)
    results = converter.convert_dataset()

    # Display results summary
    console.print()
    _display_results(results)


@main.command()
@click.argument("mha_file", type=click.Path(exists=True, path_type=Path))
def inspect(mha_file: Path):
    """Inspect an MHA file and display its properties.

    MHA_FILE: Path to the MHA file to inspect
    """
    image = sitk.ReadImage(str(mha_file))
    array = sitk.GetArrayFromImage(image)

    table = Table(title=f"MHA File: {mha_file.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Size (x, y, z)", str(image.GetSize()))
    table.add_row("Spacing (mm)", f"{image.GetSpacing()}")
    table.add_row("Origin", f"{image.GetOrigin()}")
    table.add_row("Pixel Type", image.GetPixelIDTypeAsString())
    table.add_row("Array Shape (z, y, x)", str(array.shape))
    table.add_row("Min Value", f"{np.min(array):.2f}")
    table.add_row("Max Value", f"{np.max(array):.2f}")
    table.add_row("Mean Value", f"{np.mean(array):.2f}")

    unique_vals = np.unique(array)
    table.add_row("Unique Values", str(len(unique_vals)))

    if len(unique_vals) <= 20:
        table.add_row("Labels", str(unique_vals.tolist()))

        # Show voxel counts per label
        for val in unique_vals:
            count = np.sum(array == val)
            pct = (count / array.size) * 100
            table.add_row(f"  Label {int(val)}", f"{count:,} voxels ({pct:.1f}%)")

    console.print(table)


@main.command()
@click.argument("dataset_dir", type=click.Path(exists=True, path_type=Path))
def validate(dataset_dir: Path):
    """Validate an nnU-Net dataset structure.

    DATASET_DIR: Path to the nnU-Net dataset directory
    """
    console.print(f"[bold blue]Validating nnU-Net dataset: {dataset_dir}[/bold blue]")
    console.print()

    result = validate_nnunet_dataset(dataset_dir)

    # Display issues by severity
    for issue in result.issues:
        if issue.severity == ValidationSeverity.ERROR:
            console.print(f"[red]ERROR:[/red] {issue.message}")
        elif issue.severity == ValidationSeverity.WARNING:
            console.print(f"[yellow]WARNING:[/yellow] {issue.message}")
        else:
            console.print(f"[dim]INFO:[/dim] {issue.message}")

        if issue.suggestion:
            console.print(f"  [dim]Suggestion: {issue.suggestion}[/dim]")

    console.print()

    if result.passed:
        console.print("[bold green]Dataset validation PASSED[/bold green]")
    else:
        console.print(f"[bold red]Dataset validation FAILED ({len(result.errors)} errors)[/bold red]")


@main.command()
@click.argument("dicom_dir", type=click.Path(exists=True, path_type=Path))
def analyze_mask(dicom_dir: Path):
    """Analyze a DICOM mask directory to understand its structure.

    Useful for debugging mask extraction issues.

    DICOM_DIR: Path to directory containing DICOM mask files
    """
    from .dicom_reader import DicomSeriesReader
    from .mask_processor import MimicsMaskProcessor

    console.print(f"[bold blue]Analyzing mask: {dicom_dir}[/bold blue]")
    console.print()

    reader = DicomSeriesReader()
    processor = MimicsMaskProcessor()

    try:
        # Read DICOM
        image = reader.read_dicom_series(dicom_dir)
        array = sitk.GetArrayFromImage(image)

        # Analyze
        stats = processor.analyze_mask_statistics(array)

        table = Table(title="Mask Analysis")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Shape (z, y, x)", str(stats["volume_shape"]))
        table.add_row("Total Voxels", f"{stats['total_voxels']:,}")
        table.add_row("Unique Values", str(stats["unique_values"]))
        table.add_row("Value Range", f"{stats['min_value']:.1f} to {stats['max_value']:.1f}")
        table.add_row("Mean Value", f"{stats['mean_value']:.1f}")
        table.add_row("", "")
        table.add_row("[bold]Detected Uniform HU[/bold]", f"{stats['detected_hu']:.1f}")
        table.add_row("Mask Voxels", f"{stats['mask_voxels']:,}")
        table.add_row("Mask Percentage", f"{stats['mask_percentage']:.2f}%")
        table.add_row("", "")
        table.add_row("[bold]Most Common Values[/bold]", "")

        for value, count in stats["most_common_values"][:5]:
            pct = (count / stats["total_voxels"]) * 100
            table.add_row(f"  Value {value}", f"{count:,} ({pct:.2f}%)")

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error analyzing mask: {e}[/red]")


def _display_results(results: list):
    """Display conversion results in a formatted table."""
    table = Table(title="Conversion Results")
    table.add_column("Patient", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Masks", style="green")
    table.add_column("Overlaps", style="yellow")
    table.add_column("Validation", style="magenta")

    success_count = 0
    for result in results:
        if result.success:
            status = "[green]Success[/green]"
            success_count += 1
        else:
            status = "[red]Failed[/red]"

        # Mask info
        if result.mask_info:
            masks = ", ".join(f"{m.folder_name}({m.label_value})" for m in result.mask_info)
            if len(masks) > 40:
                masks = masks[:37] + "..."
        else:
            masks = "-"

        # Overlap info
        if result.overlaps:
            overlaps = f"{len(result.overlaps)} overlap(s)"
        else:
            overlaps = "None"

        # Validation info
        if result.validation_result:
            errors = len(result.validation_result.errors)
            warnings = len(result.validation_result.warnings)
            if errors > 0:
                validation = f"[red]{errors}E[/red] {warnings}W"
            elif warnings > 0:
                validation = f"[yellow]{warnings}W[/yellow]"
            else:
                validation = "[green]OK[/green]"
        else:
            validation = "-"

        table.add_row(result.patient_id, status, masks, overlaps, validation)

    console.print(table)
    console.print(f"\n[bold]Total: {success_count}/{len(results)} successful[/bold]")


if __name__ == "__main__":
    main()
