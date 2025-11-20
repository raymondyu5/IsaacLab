#!/bin/bash

# Script to find large files tracked by git
# Usage: ./find_large_files.sh [number_of_results]

# Default to showing top 20 largest files
NUM_RESULTS=${1:-20}

echo "Finding the largest files tracked by git..."
echo "Showing top $NUM_RESULTS results..."
echo ""

# Get all tracked files, get their sizes, sort by size (descending), and show top N
git ls-files | while read file; do
    if [ -f "$file" ]; then
        size=$(du -b "$file" 2>/dev/null | cut -f1)
        if [ -n "$size" ]; then
            echo "$size $file"
        fi
    fi
done | sort -rn | head -n "$NUM_RESULTS" | while read size file; do
    # Convert bytes to human readable format
    size_human=$(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size}B")
    printf "%12s  %s\n" "$size_human" "$file"
done

echo ""
echo "Done!"

