#!/usr/bin/env python3
"""
Merge NanoLang source file with all its imports into a single file.
Handles circular dependencies and prevents duplicates.
"""

import sys
import re
import os
from collections import OrderedDict

def extract_imports(filepath):
    """Extract all import statements from a file."""
    imports = []
    if not os.path.exists(filepath):
        return imports
    
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r'^import\s+"([^"]+)"', line)
            if match:
                imports.append(match.group(1))
    return imports

def build_dependency_graph(root_file):
    """Build a dependency graph for all imports."""
    graph = OrderedDict()
    to_process = [root_file]
    processed = set()
    
    while to_process:
        current = to_process.pop(0)
        if current in processed:
            continue
        
        processed.add(current)
        imports = extract_imports(current)
        graph[current] = imports
        
        # Add new imports to process queue
        for imp in imports:
            if imp not in processed:
                to_process.append(imp)
    
    return graph

def topological_sort(graph):
    """Topologically sort files by dependencies - dependencies come FIRST."""
    # Build dependency count for each file
    visited = set()
    result = []
    
    def visit(node):
        if node in visited:
            return
        visited.add(node)
        
        # Visit dependencies first
        if node in graph:
            for dep in graph[node]:
                visit(dep)
        
        # Then add this node
        result.append(node)
    
    # Start from all root nodes
    for node in graph:
        visit(node)
    
    return result

def merge_files_in_order(files):
    """Merge files in dependency order, emitting each exactly once."""
    result = []
    emitted = set()
    
    for filepath in files:
        if filepath in emitted:
            continue
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}", file=sys.stderr)
            continue
        
        emitted.add(filepath)
        
        with open(filepath, 'r') as f:
            for line in f:
                # Skip import statements (they're already resolved)
                if re.match(r'^import\s+"([^"]+)"', line):
                    continue
                
                # Strip 'pub' keyword from definitions
                line = re.sub(r'^pub fn ', 'fn ', line)
                line = re.sub(r'^pub struct ', 'struct ', line)
                line = re.sub(r'^pub enum ', 'enum ', line)
                line = re.sub(r'^pub union ', 'union ', line)
                
                result.append(line)
    
    return ''.join(result)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: merge_imports.py <input.nano>", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Build dependency graph
    graph = build_dependency_graph(input_file)
    
    # Topologically sort files
    sorted_files = topological_sort(graph)
    
    # Debug output
    print(f"# Merging {len(sorted_files)} files in dependency order:", file=sys.stderr)
    for f in sorted_files:
        print(f"#   {f}", file=sys.stderr)
    
    # Merge files in order
    merged = merge_files_in_order(sorted_files)
    print(merged, end='')

