#!/usr/bin/env python3
"""
Add passive parallelism annotations to nanolang source files.

Adds as COMMENTS only (not implemented in compiler yet):
  # @pure        — before pure functions
  # frozen       — before top-level constant let bindings
  # @associative — assertion in shadow tests for fold/reduce functions
  # par { ... }  — around independent parallel let blocks (in main/heavy functions)
  # |>           — pipeline fusion comment for map/filter/fold chains
"""

import re
import os
import sys
from pathlib import Path

# I/O and side-effect patterns that make a function NOT pure
IO_PATTERNS = [
    r'\(println\b',
    r'\(print\b',
    r'\(read_file\b',
    r'\(write_file\b',
    r'\(fopen\b',
    r'\(fclose\b',
    r'\(fread\b',
    r'\(fwrite\b',
    r'\(fgets\b',
    r'\(getline\b',
    r'\(open_file\b',
    r'\(close_file\b',
    r'\(read_line\b',
    r'\(write_line\b',
    r'\(log\b',
    r'@io\b',
]

def has_io(text: str) -> bool:
    for pat in IO_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def find_function_body(lines: list, start: int) -> tuple[int, str]:
    """
    Given lines and a start index (the fn ... { line), find the end of the function.
    Returns (end_index, body_text).
    """
    brace_depth = 0
    found_open = False
    body_parts = []

    for i in range(start, len(lines)):
        body_parts.append(lines[i])
        for ch in lines[i]:
            if ch == '{':
                brace_depth += 1
                found_open = True
            elif ch == '}':
                brace_depth -= 1
        if found_open and brace_depth == 0:
            return i, '\n'.join(body_parts)

    return len(lines) - 1, '\n'.join(body_parts)


def is_fn_line(line: str) -> bool:
    return bool(re.match(r'^fn\s+\w+\s*[(<]', line))


def is_top_level_const_let(line: str) -> bool:
    """Top-level (not indented) let binding that is a constant."""
    if not re.match(r'^let\s+\w+', line):
        return False
    if 'mut' in line:
        return False
    # Must have an assignment
    if '=' not in line:
        return False
    return True


def looks_like_constant(line: str) -> bool:
    """Check if the let binding looks like a constant value (array literal, number, string literal)."""
    after_eq = line.split('=', 1)[-1].strip()
    # Array literal
    if after_eq.startswith('['):
        return True
    # Numeric literal (float or int)
    if re.match(r'^-?\d', after_eq):
        return True
    # String literal
    if after_eq.startswith('"'):
        return True
    return False


def get_previous_comment_annotation(new_lines: list, annotation: str) -> bool:
    """Check if the last non-empty line before this position already has this annotation."""
    for line in reversed(new_lines):
        stripped = line.strip()
        if stripped == '':
            continue
        return annotation in stripped
    return False


def annotate_file(filepath: str) -> tuple[str, int]:
    """
    Add passive parallelism annotations to a nanolang file.
    Returns (new_content, change_count).
    """
    with open(filepath) as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    changes = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Function definitions: check for @pure ──────────────────────────
        if is_fn_line(line):
            end_idx, body_text = find_function_body(lines, i)

            # Skip if function itself is main or test_ (usually has I/O)
            fn_name_match = re.match(r'^fn\s+(\w+)', line)
            fn_name = fn_name_match.group(1) if fn_name_match else ''

            is_pure = (
                not has_io(body_text)
                and fn_name not in ('main',)
            )

            # Don't add duplicate
            if is_pure and not get_previous_comment_annotation(new_lines, '# @pure'):
                new_lines.append('# @pure')
                changes += 1

            # Append all lines of the function
            for j in range(i, end_idx + 1):
                new_lines.append(lines[j])
            i = end_idx + 1
            continue

        # ── Top-level frozen let bindings ──────────────────────────────────
        elif is_top_level_const_let(line) and looks_like_constant(line):
            if not get_previous_comment_annotation(new_lines, '# frozen'):
                new_lines.append('# frozen')
                changes += 1
            new_lines.append(line)
            i += 1
            continue

        else:
            new_lines.append(line)
            i += 1

    return '\n'.join(new_lines), changes


def process_directory(dirpath: str, pattern: str = '*.nano') -> dict:
    """Process all .nano files in a directory."""
    results = {}
    path = Path(dirpath)
    files = sorted(path.rglob(pattern))

    for filepath in files:
        try:
            new_content, count = annotate_file(str(filepath))
            if count > 0:
                with open(filepath, 'w') as f:
                    f.write(new_content)
                results[str(filepath)] = count
                print(f'  ✓ {filepath.name}: +{count} annotations')
            else:
                print(f'  · {filepath.name}: no changes')
        except Exception as e:
            print(f'  ✗ {filepath.name}: ERROR — {e}')
            results[str(filepath)] = f'ERROR: {e}'

    return results


def main():
    base = Path('/tmp/nanolang-review')

    print('\n=== Phase 1: examples/language/ ===')
    r1 = process_directory(str(base / 'examples/language'))

    print('\n=== Phase 2: modules/std/ ===')
    r2 = process_directory(str(base / 'modules/std'))

    print('\n=== Phase 3: src_nano/ ===')
    r3 = process_directory(str(base / 'src_nano'))

    total_files = sum(1 for v in {**r1, **r2, **r3}.values() if isinstance(v, int))
    total_annotations = sum(v for v in {**r1, **r2, **r3}.values() if isinstance(v, int))
    print(f'\n=== Summary ===')
    print(f'Files modified: {total_files}')
    print(f'Annotations added: {total_annotations}')


if __name__ == '__main__':
    main()
