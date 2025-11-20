#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# Find installed package location

source_path = Path.home() / "apptainer" / "IsaacLab" / "source" / "config"
installed_path = Path.home() / "IsaacLab" / "source" / "config"

# Copy only .py files recursively

for py_file in source_path.rglob("*.py"):
    relative_path = py_file.relative_to(source_path)
    dest_file = installed_path / relative_path
    
    # Create parent directory if needed
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the file
    shutil.copy2(py_file, dest_file)
    print(f"Copied: {relative_path}")

print(f"\nDone! Replaced Python files in: {installed_path}")