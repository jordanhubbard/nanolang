#!/usr/bin/env python3
"""I require explicit actual-GPU execution; each fault case is a new process."""
try:
    from tests.sanitizer_options import asan_options
except ModuleNotFoundError:
    from sanitizer_options import asan_options
import argparse
import os
from pathlib import Path
import subprocess

CASES = (['normal', 'contexts', 'admit-loader', 'admit-host0', 'admit-host1']
         + ['admit-symbol' + str(i) for i in range(1, 17)]
         + ['admit-' + op for op in ['init', 'version', 'count', 'device', 'name', 'uuid', 'create', 'current']]
         + ['create-' + case for case in ['post', 'popbefore', 'popafter', 'query']]
         + ['allocbefore', 'allocafter', 'rollback', 'pop-before', 'pop-after',
            'freebefore', 'freeafter', 'closepush', 'destroybefore', 'destroyafter',
            'staging', 'generation', 'readbefore', 'readafter', 'readsync',
            'writebefore', 'writeafter', 'writesync', 'loader-close'])

def commands(compiler, output, sanitizers=False):
    common = [compiler, '-std=c11', '-D_DEFAULT_SOURCE', '-Isrc', '-O1', '-g',
              '-Wall', '-Wextra', '-Werror', '-pedantic']
    if sanitizers:
        common += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    yield 'instrumented-build', common + ['tests/test_nsi_gpu.c', '-ldl', '-o', str(output / 'instrumented')]
    yield 'linked-build', common + ['tests/test_nsi_gpu_linked.c', 'src/nsi_gpu.c', 'src/nsi_cap.c', '-ldl', '-o', str(output / 'linked')]
    yield 'linked-real-run', [str(output / 'linked')]
    for case in CASES:
        yield case, [str(output / 'instrumented'), case]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-gpu', action='store_true', required=True)
    parser.add_argument('--compiler', default='cc')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sanitizers', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, command in commands(args.compiler, args.output.resolve(), args.sanitizers):
        print('CASE', name, flush=True)
        subprocess.run(command, check=True, timeout=180, env={**os.environ,
                       'ASAN_OPTIONS': asan_options("halt_on_error=1"),
                       'UBSAN_OPTIONS': 'halt_on_error=1:print_stacktrace=1'})
