#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 <input_directory> <output_file>"
    echo "Concatenates all .jsonl files in the specified directory and its subdirectories"
    exit 1
}

# Check if correct number of arguments is provided
if [ $# -ne 2 ]; then
    usage
fi

# Input directory and output file
INPUT_DIR="$1"
OUTPUT_FILE="$2"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Create/truncate the output file
> "$OUTPUT_FILE"

# Find and concatenate all .jsonl files
find "$INPUT_DIR" -type f -name "*.jsonl" -print0 | while IFS= read -r -d '' file; do
    # Append each file to the output, adding a newline if the file doesn't end with a newline
    cat "$file" >> "$OUTPUT_FILE"
    # Ensure the last line ends with a newline if it doesn't already
    [[ $(tail -c1 "$file") && $(tail -c1 "$file") != $'\n' ]] && echo "" >> "$OUTPUT_FILE"
done

echo "Successfully concatenated all .jsonl files from '$INPUT_DIR' to '$OUTPUT_FILE'"