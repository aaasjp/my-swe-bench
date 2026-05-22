#!/bin/bash
# Script to create text datasets from SWE-bench

# Configuration - modify these as needed
DATASET_NAME="princeton-nlp/SWE-bench_Verified"
OUTPUT_DIR="./base_datasets"
SPLITS="test"
PROMPT_STYLE="style-3"
FILE_SOURCE="oracle"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create text dataset
python -m swebench.inference.make_datasets.create_text_dataset \
    --dataset_name_or_path "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --splits "$SPLITS" \
    --prompt_style "$PROMPT_STYLE" \
    --file_source "$FILE_SOURCE"

echo "Text dataset created successfully in $OUTPUT_DIR"