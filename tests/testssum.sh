#!/bin/bash

output_file="output"

# Clear the output file if it exists
> "$output_file"

# Process run_integration_tests.py first
echo "### File: run_integration_tests.py" >> "$output_file"
cat run_integration_tests.py >> "$output_file"
echo "" >> "$output_file" # Add a newline for separation

# Process all files starting with 'test'
for file in test*; do
    if [ -f "$file" ]; then # Ensure it's a regular file
        echo "### File: $file" >> "$output_file"
        cat "$file" >> "$output_file"
        echo "" >> "$output_file" # Add a newline for separation
    fi
done

echo "Concatenated files into '$output_file' with headers."
