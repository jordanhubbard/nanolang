#!/usr/bin/env python3
"""I run my fast pre-push gate: strict compiles of my product tools with every available compiler.

I read the compile commands from a Makefile dry run, so I use the real flags and cannot drift from the
build. I discard the objects, run the compiles in parallel and report only failures. A GCC compile with
optimization finds the maybe-uninitialized and unused-but-set errors that cost a hosted round trip."""
import argparse
import concurrent.futures
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['nano_virt', 'nano_vm', 'nvm2c', 'nanoisa_dump', 'nvm2c-runtime']
DROP = {'-MMD', '-MP'}


def detect_compilers():
    found = {}
    for name in ('clang', 'cc'):
        path = shutil.which(name)
        if path and 'clang' in subprocess.run([path, '--version'], capture_output=True, text=True).stdout.lower():
            found['clang'] = path
            break
    versions = sorted((p for p in (shutil.which(f'gcc-{n}') for n in range(9, 30)) if p), reverse=True)
    gcc = versions[0] if versions else shutil.which('gcc')
    if gcc and 'gcc' in subprocess.run([gcc, '--version'], capture_output=True, text=True).stdout.lower() and \
            'clang' not in subprocess.run([gcc, '--version'], capture_output=True, text=True).stdout.lower():
        found['gcc'] = gcc
    return found


def compile_commands(compiler):
    result = subprocess.run(['make', '-n', '-B', f'CC={compiler}', *TARGETS], cwd=ROOT,
                            capture_output=True, text=True, timeout=300)
    commands = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith(compiler) or ' -c ' not in line or '&&' in line or ';' in line:
            continue
        try:
            words = shlex.split(line)
        except ValueError:
            continue
        cleaned, skip = [], False
        for word in words:
            if skip:
                skip = False
                continue
            if word in DROP:
                continue
            if word in ('-MF', '-MT', '-o'):
                skip = True
                continue
            cleaned.append(word)
        # I stop before assembly (-S): I need the compiler's diagnostics, and a host assembler can
        # reject valid code (Darwin GCC reports an unrelated relocation error for runtime/gc.c).
        commands.append(['-S' if word == '-c' else word for word in cleaned] + ['-o', os.devnull])
    return commands


def run(command):
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return command, result.returncode, result.stderr or result.stdout, time.monotonic() - started


def source_of(command):
    return next((word for word in command if re.search(r'\.c(pp)?$', word)), '?')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    parser.add_argument('--compilers', default='', help='I limit the run to a comma-separated subset of gcc,clang.')
    parser.add_argument('--optimize', default='-O2', help='I add this level so uninitialized analysis runs.')
    args = parser.parse_args()
    available = detect_compilers()
    wanted = [name for name in (args.compilers.split(',') if args.compilers else available) if name]
    missing = [name for name in wanted if name not in available]
    if missing:
        print('I cannot find: ' + ', '.join(missing), file=sys.stderr)
        return 2
    failed = 0
    for name in wanted:
        started = time.monotonic()
        commands = compile_commands(available[name])
        if not commands:
            print(f'{name}: I found no compile commands in the dry run.', file=sys.stderr)
            return 2
        # I replace any existing optimization level so GCC analyzes uninitialized use.
        commands = [[word for word in command if not re.fullmatch(r'-O[0-3sg]?', word)] + [args.optimize]
                    for command in commands]
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            results = list(pool.map(run, commands))
        bad = [item for item in results if item[1] != 0]
        failed += len(bad)
        print(f'{name} ({available[name]}): {len(results)} compiles, {len(bad)} failed, '
              f'{time.monotonic() - started:.1f}s')
        for command, _, output, _ in bad[:20]:
            errors = [line for line in output.splitlines() if 'error' in line][:3]
            print(f'  FAIL {source_of(command)}')
            for line in errors:
                print('    ' + line[:200])
    print('fast gate: ' + ('FAILED' if failed else 'ok'))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
