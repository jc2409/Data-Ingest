#!/bin/bash

# Script to chunk all documents in dataset/res/ directory
# Uses single-document mode for better control and error handling

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Chunking All Documents"
echo "=========================================="

# Directory containing source JSON files
RES_DIR="dataset/res"

# Check if directory exists
if [ ! -d "$RES_DIR" ]; then
    echo -e "${RED}Error: Directory $RES_DIR not found!${NC}"
    exit 1
fi

# Count total files
total_files=$(ls -1 "$RES_DIR"/*.json 2>/dev/null | wc -l)

if [ "$total_files" -eq 0 ]; then
    echo -e "${RED}Error: No JSON files found in $RES_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Found $total_files JSON files to process${NC}"
echo ""

# Counter for progress
processed=0
successful=0
failed=0

# Array to store failed files
declare -a failed_files

# Process each JSON file
for file in "$RES_DIR"/*.json; do
    ((processed++))

    # Get filename without path
    filename=$(basename "$file")

    echo "=========================================="
    echo -e "${YELLOW}[$processed/$total_files] Processing: $filename${NC}"
    echo "=========================================="

    # Run chunking with single-document mode
    if uv run python src/contextual_chunking.py -s "$file"; then
        ((successful++))
        echo -e "${GREEN}✓ Successfully processed: $filename${NC}"
    else
        ((failed++))
        failed_files+=("$filename")
        echo -e "${RED}✗ Failed to process: $filename${NC}"
    fi

    echo ""
done

# Summary
echo "=========================================="
echo "PROCESSING COMPLETE"
echo "=========================================="
echo -e "Total files:      $total_files"
echo -e "${GREEN}Successful:       $successful${NC}"

if [ "$failed" -gt 0 ]; then
    echo -e "${RED}Failed:           $failed${NC}"
    echo ""
    echo "Failed files:"
    for failed_file in "${failed_files[@]}"; do
        echo -e "  ${RED}✗ $failed_file${NC}"
    done
else
    echo -e "${GREEN}Failed:           0${NC}"
    echo -e "\n${GREEN}All files processed successfully! 🎉${NC}"
fi

echo "=========================================="

# Exit with error code if any failed
if [ "$failed" -gt 0 ]; then
    exit 1
fi

exit 0
