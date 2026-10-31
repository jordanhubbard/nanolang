#!/usr/bin/env python3
"""I inventory my native build graph for Literate AI: typed units, header closure, outputs and toolchain.

I read the real commands from a `make -n -B build` dry run, so I cannot drift from the build I ship. Nothing
is built or executed except `cc -MM -MG` header scans and tool version probes. I write one deterministic
host-qualified JSON document. A missing generated header is reported as generated, never invented. I check
that every output has exactly one producer. This is an inventory, not a qualification receipt."""
import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DROP = {'-MMD', '-MP'}


def sh(args, **kw):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, **kw)


def dry_run(target):
    result = sh(['make', '-n', '-B', target], timeout=600)
    if result.returncode != 0:
        sys.exit(f'dry run of {target} failed: {result.stderr[-400:]}')
    return result.stdout.replace('\\\n', ' ').splitlines()


def tracked():
    return set(sh(['git', 'ls-files', '-z']).stdout.split('\0'))


def tokens(line):
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return None


def classify(lines):
    units, links, archives, generators = [], [], [], []
    unclassified = []
    for raw in lines:
        line = raw.strip()
        tok = tokens(line) if line else None
        if not tok:
            if re.match(r'(cc|clang|gcc)\b', line):
                unclassified.append(line[:200])
            continue
        head = Path(tok[0]).name
        if head in ('cc', 'clang', 'gcc') or re.fullmatch(r'(gcc|clang)-\d+', head):
            out = tok[tok.index('-o') + 1] if '-o' in tok else None
            if out is None:
                unclassified.append(line[:200])
                continue
            if '-c' in tok:
                src = tok[tok.index('-c') + 1]
                units.append({'compiler': tok[0], 'source': src, 'object': out,
                              'flags': [t for t in tok[1:] if t not in DROP and t not in ('-c', src, '-o', out)]})
            else:
                links.append({'output': out, 'inputs': [t for t in tok[1:] if t.endswith(('.o', '.a'))],
                              'flags': [t for t in tok[1:] if t.startswith(('-l', '-L', '-f', '-W', '-O', '-std', '-g'))]})
        elif head == 'ar' and len(tok) >= 4:
            archives.append({'output': tok[2], 'inputs': tok[3:]})
        elif head in ('python3', 'python') or tok[0].endswith('/python3'):
            script = next((t for t in tok[1:] if t.endswith('.py')), None)
            if script:
                generators.append({'tool': 'python3', 'script': script, 'arguments': [t for t in tok[1:] if t != script]})
        elif tok[0].startswith('./bin/') and len(tok) == 1:
            generators.append({'tool': tok[0][2:], 'script': None, 'arguments': []})
    return units, links, archives, generators, unclassified


def header_closure(unit, git_files):
    flags = [f for f in unit['flags'] if f.startswith(('-I', '-D', '-std', '-isystem'))]
    result = sh([unit['compiler'], *flags, '-MM', '-MG', unit['source']])
    text = result.stdout.replace('\\\n', ' ')
    names = text.split(':', 1)[1].split() if ':' in text else []
    deps = sorted({os.path.normpath(n) for n in names if os.path.normpath(n) != unit['source']})
    repo = [d for d in deps if not d.startswith('/')]
    return {
        'tracked': [d for d in repo if d in git_files],
        'generated_or_absent': [d for d in repo if d not in git_files],
        'system': [d for d in deps if d.startswith('/')],
        'scan_ok': result.returncode == 0,
    }


def toolchain(compiler):
    version = sh([compiler, '--version']).stdout.splitlines()
    sysroot = sh(['xcrun', '--show-sdk-path']).stdout.strip() if platform.system() == 'Darwin' else None
    path = sh(['which', compiler]).stdout.strip()
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
    return {'compiler': compiler, 'path': path, 'sha256': digest, 'banner': version[0] if version else None,
            'sysroot': sysroot, 'host': f'{platform.system().lower()}-{platform.machine()}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default='build')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true', help='exit 1 on duplicate producers or failed scans')
    args = parser.parse_args()
    git_files = tracked()
    units, links, archives, generators, unclassified = classify(dry_run(args.target))
    for unit in units:
        unit['headers'] = header_closure(unit, git_files)
        unit['source_tracked'] = unit['source'] in git_files
    owners = {}
    for kind, items in (('compile', units), ('link', links), ('archive', archives)):
        for item in items:
            owners.setdefault(item.get('object') or item['output'], []).append(kind)
    duplicates = {k: v for k, v in owners.items() if len(v) > 1}
    scan_failures = [u['source'] for u in units if not u['headers']['scan_ok']]
    nano = sorted(p for p in git_files if p.startswith('src_nano/') and p.endswith('.nano'))
    compilers = sorted({u['compiler'] for u in units})
    report = {
        'schema': 'nanolang/native-unit-inventory@1',
        'authority': 'inventory-only: not a Literate AI receipt, not qualification',
        'dry_run_target': args.target,
        'toolchains': [toolchain(c) for c in compilers],
        'summary': {'compile_units': len(units), 'links': len(links), 'archives': len(archives),
                    'generator_invocations': len(generators), 'nano_sources': len(nano),
                    'duplicate_output_producers': len(duplicates), 'unclassified_compiler_lines': len(unclassified), 'header_scan_failures': len(scan_failures),
                    'units_needing_generated_headers': sum(1 for u in units if u['headers']['generated_or_absent'])},
        'compile_units': sorted(units, key=lambda u: u['object']),
        'links': sorted(links, key=lambda l: l['output']),
        'archives': sorted(archives, key=lambda a: a['output']),
        'generators': generators,
        'nano_sources': nano,
        'duplicate_output_producers': duplicates,
        'header_scan_failures': scan_failures,
        'unclassified_compiler_lines': unclassified,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    print(json.dumps(report['summary'], sort_keys=True), file=sys.stderr)
    return 1 if args.check and (duplicates or scan_failures) else 0


if __name__ == '__main__':
    sys.exit(main())
