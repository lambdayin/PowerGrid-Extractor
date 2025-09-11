#!/bin/bash
# Example script for running power transmission objects extraction
# Based on Zhang et al., Remote Sensing 2019

set -e  # Exit on error

echo "Power Transmission Objects Extraction - Example Run"
echo "=================================================="

# Check if input LAS file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_las_file> [output_directory]"
    echo "Example: $0 data/power_corridor.las ./results"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="${2:-./results}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run extraction with default parameters
echo "Running power transmission objects extraction..."
echo "Using paper-specified parameters:"
echo "  - 2D grid size: 5.0m x 5.0m"
echo "  - 3D voxel size: 0.5m³"
echo "  - Minimum height gap: 8.0m"
echo "  - Linear threshold: 0.6"
echo ""

python -m corridor_seg \
    --input "$INPUT_FILE" \
    --outdir "$OUTPUT_DIR" \
    --config examples/default_config.yaml \
    --visualize \
    --log-level INFO

echo ""
echo "Extraction complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "  - powerlines.las: Extracted power line points"
echo "  - towers.las: Extracted tower points"
echo "  - processing_report.txt: Detailed processing report"

# Check if results exist
if [ -f "$OUTPUT_DIR/powerlines.las" ]; then
    echo "✓ Power lines successfully extracted"
else
    echo "⚠ No power lines found"
fi

if [ -f "$OUTPUT_DIR/towers.las" ]; then
    echo "✓ Towers successfully extracted"
else
    echo "⚠ No towers found"
fi

echo ""
echo "To run with custom parameters, edit examples/default_config.yaml or use command line options:"
echo "  python -m corridor_seg --input your.las --outdir ./out --grid-size 10.0 --min-height-gap 5.0"