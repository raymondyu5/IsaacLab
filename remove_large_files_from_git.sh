#!/bin/bash

# Script to remove large files from git tracking and add them to .gitignore
# Usage: ./remove_large_files_from_git.sh [file1] [file2] ... OR run interactively

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

GITIGNORE=".gitignore"

# Function to add pattern to .gitignore if not already present
add_to_gitignore() {
    local pattern="$1"
    
    # Check if pattern already exists in .gitignore
    if grep -Fxq "$pattern" "$GITIGNORE" 2>/dev/null; then
        echo -e "${YELLOW}Pattern '$pattern' already exists in .gitignore${NC}"
        return 0
    fi
    
    # Add pattern to .gitignore
    echo "$pattern" >> "$GITIGNORE"
    echo -e "${GREEN}Added '$pattern' to .gitignore${NC}"
}

# Function to remove file from git tracking
remove_from_git() {
    local file="$1"
    
    if [ ! -f "$file" ] && [ ! -d "$file" ]; then
        echo -e "${RED}Error: '$file' does not exist${NC}"
        return 1
    fi
    
    # Check if file is tracked by git
    if ! git ls-files --error-unmatch "$file" > /dev/null 2>&1; then
        echo -e "${YELLOW}'$file' is not tracked by git${NC}"
        return 0
    fi
    
    # Remove from git index (but keep local file)
    git rm --cached -r "$file"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Removed '$file' from git tracking${NC}"
        
        # Add to .gitignore
        # If it's a specific file, add the exact path
        # If it's a directory, add the directory pattern
        if [ -d "$file" ]; then
            add_to_gitignore "$file/"
        else
            add_to_gitignore "$file"
        fi
        return 0
    else
        echo -e "${RED}Failed to remove '$file' from git${NC}"
        return 1
    fi
}

# Interactive mode: show large files and let user select
if [ $# -eq 0 ]; then
    echo "Finding large files tracked by git..."
    echo ""
    
    # Get top 50 largest files
    large_files=$(git ls-files | while read file; do
        if [ -f "$file" ]; then
            size=$(du -b "$file" 2>/dev/null | cut -f1)
            if [ -n "$size" ] && [ "$size" -gt 1048576 ]; then  # Only files > 1MB
                echo "$size $file"
            fi
        fi
    done | sort -rn | head -n 50)
    
    if [ -z "$large_files" ]; then
        echo "No large files found (>1MB)"
        exit 0
    fi
    
    echo "Large files found (>1MB):"
    echo ""
    count=1
    declare -a file_array
    while IFS= read -r line; do
        if [ -n "$line" ]; then
            size=$(echo "$line" | cut -d' ' -f1)
            file=$(echo "$line" | cut -d' ' -f2-)
            size_human=$(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "${size}B")
            printf "%3d. %12s  %s\n" "$count" "$size_human" "$file"
            file_array[$count]="$file"
            ((count++))
        fi
    done <<< "$large_files"
    
    echo ""
    echo "Enter file numbers to remove (space-separated, or 'all' for all, or 'q' to quit):"
    read -r selection
    
    if [ "$selection" = "q" ] || [ "$selection" = "Q" ]; then
        exit 0
    fi
    
    if [ "$selection" = "all" ] || [ "$selection" = "ALL" ]; then
        for file in "${file_array[@]}"; do
            if [ -n "$file" ]; then
                remove_from_git "$file"
            fi
        done
    else
        for num in $selection; do
            if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -lt "$count" ]; then
                remove_from_git "${file_array[$num]}"
            else
                echo -e "${RED}Invalid selection: $num${NC}"
            fi
        done
    fi
else
    # Non-interactive mode: remove specified files
    for file in "$@"; do
        remove_from_git "$file"
    done
fi

echo ""
echo -e "${GREEN}Done! Remember to commit the changes:${NC}"
echo "  git add .gitignore"
echo "  git commit -m 'Remove large files from tracking'"

