#!/usr/bin/env python3
"""
Shadow Test Audit - Comprehensive Analysis
"""

import os
import re
from collections import defaultdict
from pathlib import Path

def extract_functions(content):
    """Extract function names from NanoLang code"""
    # Match: fn name(...) or pub fn name(...)
    func_pattern = r'^(?:pub\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    functions = set()
    for line in content.split('\n'):
        match = re.match(func_pattern, line)
        if match:
            functions.add(match.group(1))
    return functions

def extract_externs(content):
    """Extract extern function names"""
    extern_pattern = r'^extern\s+fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    externs = set()
    for line in content.split('\n'):
        match = re.match(extern_pattern, line)
        if match:
            externs.add(match.group(1))
    return externs

def extract_shadows(content):
    """Extract shadow test names"""
    shadow_pattern = r'^shadow\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    shadows = set()
    for line in content.split('\n'):
        match = re.match(shadow_pattern, line)
        if match:
            shadows.add(match.group(1))
    return shadows

def should_exclude(func_name):
    """Check if function should be excluded from shadow test requirement"""
    excluded = {'main', 'init', 'cleanup', 'setup', 'teardown'}
    return func_name in excluded

def categorize_path(path):
    """Categorize file by directory"""
    path_str = str(path)
    if 'examples/' in path_str:
        return 'examples'
    elif 'modules/' in path_str:
        return 'modules'
    elif 'tests/' in path_str:
        return 'tests'
    elif 'stdlib/' in path_str or 'std/' in path_str:
        return 'stdlib'
    elif 'src_nano/' in path_str:
        return 'src_nano'
    elif 'tools/' in path_str or 'scripts/' in path_str:
        return 'tools'
    else:
        return 'other'

def analyze_file(filepath):
    """Analyze a single .nano file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return None
    
    functions = extract_functions(content)
    externs = extract_externs(content)
    shadows = extract_shadows(content)
    
    # Remove externs and excluded functions
    testable_functions = functions - externs
    testable_functions = {f for f in testable_functions if not should_exclude(f)}
    
    missing = testable_functions - shadows
    
    return {
        'path': filepath,
        'functions': testable_functions,
        'shadows': shadows,
        'missing': missing,
        'category': categorize_path(filepath)
    }

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║        COMPREHENSIVE SHADOW TEST AUDIT                        ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Find all .nano files
    nano_files = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if 'build_bootstrap' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.nano'):
                nano_files.append(os.path.join(root, file))
    
    # Analyze all files
    results = []
    for filepath in sorted(nano_files):
        result = analyze_file(filepath)
        if result and (result['functions'] or result['shadows']):
            results.append(result)
    
    # Aggregate by category
    stats = defaultdict(lambda: {'functions': 0, 'shadows': 0, 'missing': 0, 'files': []})
    
    for result in results:
        cat = result['category']
        stats[cat]['functions'] += len(result['functions'])
        stats[cat]['shadows'] += len(result['shadows'])
        stats[cat]['missing'] += len(result['missing'])
        if result['missing']:
            stats[cat]['files'].append(result)
    
    # Print summary by category
    print("═══════════════════════════════════════════════════════════════")
    print("SUMMARY BY CATEGORY:")
    print("═══════════════════════════════════════════════════════════════")
    print()
    
    total_funcs = 0
    total_shadows = 0
    total_missing = 0
    
    categories = ['examples', 'modules', 'tests', 'stdlib', 'src_nano', 'tools', 'other']
    
    for cat in categories:
        if cat not in stats:
            continue
        s = stats[cat]
        total_funcs += s['functions']
        total_shadows += s['shadows']
        total_missing += s['missing']
        
        coverage = (s['shadows'] * 100 // s['functions']) if s['functions'] > 0 else 0
        
        emoji = {
            'examples': '📚',
            'modules': '🔧',
            'tests': '✅',
            'stdlib': '📦',
            'src_nano': '🏗️ ',
            'tools': '🛠️ ',
            'other': '📁'
        }.get(cat, '📁')
        
        print(f"{emoji} {cat.upper()}:")
        print(f"   Functions:    {s['functions']}")
        print(f"   With shadows: {s['shadows']}")
        print(f"   Missing:      {s['missing']}")
        print(f"   Coverage:     {coverage}%")
        print()
    
    print("═══════════════════════════════════════════════════════════════")
    print("OVERALL TOTALS:")
    print("═══════════════════════════════════════════════════════════════")
    print()
    overall_coverage = (total_shadows * 100 // total_funcs) if total_funcs > 0 else 0
    print(f"Total functions:          {total_funcs}")
    print(f"Functions with shadows:   {total_shadows}")
    print(f"Functions missing shadows: {total_missing}")
    print(f"Overall coverage:         {overall_coverage}%")
    print()
    
    # Show files with most missing tests (top 20)
    print("═══════════════════════════════════════════════════════════════")
    print("TOP 20 FILES WITH MOST MISSING SHADOW TESTS:")
    print("═══════════════════════════════════════════════════════════════")
    print()
    
    all_files_with_missing = []
    for result in results:
        if result['missing']:
            all_files_with_missing.append((len(result['missing']), result))
    
    all_files_with_missing.sort(key=lambda x: x[0], reverse=True)
    
    for count, result in all_files_with_missing[:20]:
        print(f"{count:3d} missing: {result['path']}")
        for func in sorted(result['missing'])[:5]:  # Show first 5
            print(f"           - {func}")
        if len(result['missing']) > 5:
            print(f"           ... and {len(result['missing']) - 5} more")
        print()
    
    print("═══════════════════════════════════════════════════════════════")
    print("RECOMMENDATIONS:")
    print("═══════════════════════════════════════════════════════════════")
    print()
    print("📦 CRITICAL (Must fix):")
    print("   - stdlib/ functions: 100% coverage required")
    print("   - modules/ public APIs: Shadow tests mandatory")
    print()
    print("⚠️  HIGH PRIORITY:")
    print("   - src_nano/ compiler: Critical for self-hosting")
    print("   - modules/ internals: Important for reliability")
    print()
    print("✅ MEDIUM PRIORITY:")
    print("   - tests/ helpers: Improves test reliability")
    print("   - examples/ logic functions: Good practice")
    print()
    print("ℹ️  OK TO SKIP:")
    print("   - I/O heavy functions (file/network/graphics)")
    print("   - main() functions")
    print("   - Simple render/draw helpers")
    print("   - FFI wrappers (tested via integration)")
    print()

if __name__ == '__main__':
    main()
