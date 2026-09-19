#!/usr/bin/env python3
"""I keep host models separate from explicitly requested actual OpenCL GPU gates."""
import argparse
import os
from pathlib import Path
import subprocess

CASES = ['loader', 'ordinary', 'arguments', 'bounds', 'exhaustion',
         'allocation-error', 'allocation-null', 'rollback', 'rollback-unknown',
         'release-unknown']

def commands(compiler, out, sanitize=False, real=False):
    # Existing POSIX dlsym casts are outside ISO-pedantic conformance; no warning suppression.
    common = [compiler, '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror']
    if sanitize:
        common += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    yield 'model-build', common + ['tests/test_opencl_identity.c', '-ldl', '-o', str(out / 'model')]
    for case in CASES:
        yield 'model-' + case, [str(out / 'model'), case]
    for kind, flags in [('cuda', []), ('unified', ['-DTEST_UNIFIED_GPU'])]:
        yield kind + '-build', common + flags + ['tests/test_gpu_array_boundary.c', '-ldl', '-o', str(out / kind)]
        yield kind + '-run', [str(out / kind)]
    for kind, flags in [('cuda', []), ('unified', ['-DTEST_UNIFIED_GPU'])]:
        binary = out / (kind + '-reader')
        yield kind + '-reader-build', common + flags + ['tests/test_gpu_source_read.c', '-ldl', '-o', str(binary)]
        for reader in (['cuda', 'opencl'] if flags else ['cuda']):
            for phase in ['normal', 'empty', 'seek', 'tell', 'rewind', 'allocation', 'short', 'error', 'close']:
                yield kind + '-reader-' + reader + '-' + phase, [str(binary), phase, reader]
    if real:
        yield 'real-build', common + ['tests/test_opencl_identity_real.c', '-ldl', '-o', str(out / 'real')]
        yield 'real-run', [str(out / 'real')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='cc')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--real-gpu', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, command in commands(args.compiler, args.output.resolve(), args.sanitize, args.real_gpu):
        print('CASE', name, command, flush=True)
        subprocess.run(command, check=True, timeout=180, env={**os.environ,
                       'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                       'UBSAN_OPTIONS': 'halt_on_error=1:print_stacktrace=1'})
