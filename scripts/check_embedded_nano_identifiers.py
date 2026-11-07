#!/usr/bin/env python3
"""I check static Python fixture literals with my actual NanoLang lexer."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = re.compile(r'\bfn\s+[A-Za-z_]\w*\s*[(<]|\blet\s+(?:mut\s+)?[A-Za-z_]\w*\s*[:=]')


def literals(path):
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines()
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)
                  and isinstance(node.body[0].value.value, str)}
    selected = []
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Constant) or id(node) in docstrings or
                not isinstance(node.value, str) or not DECLARATION.search(node.value)):
            continue
        prior = lines[node.lineno - 2] if node.lineno > 1 else ''
        marker = re.fullmatch(r'\s*# nano-fixture: expected-reserved-name=([A-Za-z_]\w*)\s*', prior)
        selected.append((path, node.lineno, node.value, marker[1] if marker else None))
    return selected


def inspect(lexer, literal):
    path, line, source, expected = literal
    result = subprocess.run([str(lexer)], input=source, capture_output=True, text=True, timeout=10)
    if result.returncode == 2:
        if expected:
            raise RuntimeError(f'I require a lexable reserved-name control at {path}:{line}.')
        # I report lexer refusals separately; their executable tests still own them.
        return [], True
    if result.returncode:
        raise RuntimeError(f'I could not inspect {path}:{line}: lexer status {result.returncode}.')
    findings = []
    for row in result.stdout.splitlines():
        nano_line, column, name = row.split('\t')
        findings.append({'path': str(path), 'literal_line': line,
                         'nano_line': int(nano_line), 'nano_column': int(column), 'name': name})
    if expected:
        if not any(item['name'] == expected for item in findings):
            raise RuntimeError(f'I require the annotated reserved name {expected!r} at {path}:{line}.')
        findings = [item for item in findings if item['name'] != expected]
    return findings, False


def scan(lexer, paths, jobs=4):
    candidates = [item for path in paths for item in literals(path)]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda item: inspect(lexer, item), candidates))
    return ([finding for findings, _ in results for finding in findings],
            len(candidates), sum(refused for _, refused in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lexer', type=Path, required=True)
    parser.add_argument('paths', nargs='*', type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT/'tests').rglob('*.py'))
    findings, checked, refused = scan(args.lexer.resolve(), paths)
    for item in findings:
        print(f"{item['path']}:{item['literal_line']}: I cannot use reserved word {item['name']!r} "
              f"as a declaration name (embedded line {item['nano_line']}, column {item['nano_column']}).")
    print(f'I lexed {checked} static fixture fragments; {len(findings)} reserved names; '
          f'{refused} lexer-refused fragments remain with their executable tests.')
    return 1 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
