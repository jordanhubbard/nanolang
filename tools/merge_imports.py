#!/usr/bin/env python3
"""
Merge NanoLang source file with all its imports into a single file.
This is a simple implementation for bootstrap purposes.
"""

import sys
import re
import os

def merge_file(filepath, visited=None):
    """Recursively merge a file and its imports."""
    if visited is None:
        visited = set()
    
    if filepath in visited:
        return ""
    
    visited.add(filepath)
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}", file=sys.stderr)
        return ""
    
    result = []
    with open(filepath, 'r') as f:
        for line in f:
            # Check for import statements
            match = re.match(r'^import\s+"([^"]+)"', line)
            if match:
                import_path = match.group(1)
                # Recursively merge imported file
                imported_content = merge_file(import_path, visited)
                result.append(imported_content)
            else:
                # Strip 'pub' keyword from definitions
                line = re.sub(r'^pub fn ', 'fn ', line)
                line = re.sub(r'^pub struct ', 'struct ', line)
                result.append(line)
    
    return ''.join(result)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: merge_imports.py <input.nano>", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    merged = merge_file(input_file)
    print(merged, end='')

