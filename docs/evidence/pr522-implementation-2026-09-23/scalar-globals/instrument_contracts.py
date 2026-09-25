"""I instrument scalar-global transport and its owned execution boundary."""
from pathlib import Path
import os
import platform
import shlex
import subprocess
import tempfile

commands = subprocess.check_output(['make', '-n', 'test-owned-scalar-global-contracts'], text=True)
link = next(shlex.split(line) for line in commands.replace('\\\n', ' ').splitlines()
            if 'tests/nanoisa/test_owned_scalar_global_contracts.c' in line)
homebrew = Path('/opt/homebrew/opt/llvm/bin/clang')
compiler = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC',
                      str(homebrew) if platform.system() == 'Darwin' and homebrew.exists() else 'clang'))
flags = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-O1', '-g']
with tempfile.TemporaryDirectory(prefix='nano-global-contracts-') as directory:
    for unit in ('ownership_contracts', 'verifier', 'affine_bytecode', 'affine_state'):
        obj = str(Path(directory) / (unit + '.o'))
        subprocess.run([*compiler, '-std=c99', '-D_GNU_SOURCE', '-Isrc', '-Isrc/nanoisa',
                        *flags, '-c', f'src/nanoisa/{unit}.c', '-o', obj], check=True)
        link = [obj if item == f'obj/nanoisa/{unit}.o' else item for item in link]
    binary = str(Path(directory) / 'contracts')
    link[link.index('-o') + 1] = binary
    subprocess.run([*compiler, *link[1:], *flags], check=True)
    subprocess.run([binary], check=True,
                   env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                        'UBSAN_OPTIONS': 'halt_on_error=1'})

    allocation_object = str(Path(directory) / 'affine_bytecode_alloc.o')
    subprocess.run([*compiler, '-std=c99', '-D_GNU_SOURCE', '-Isrc', '-Isrc/nanoisa', *flags,
                    '-Dmalloc=global_flow_test_malloc', '-Dcalloc=global_flow_test_calloc',
                    '-Drealloc=global_flow_test_realloc', '-c', 'src/nanoisa/affine_bytecode.c',
                    '-o', allocation_object], check=True)
    allocation_link = [allocation_object if item == str(Path(directory) / 'affine_bytecode.o')
                       else item for item in link]
    subprocess.run([*compiler, *allocation_link[1:], '-DGLOBAL_FLOW_ALLOCATION_TEST', *flags], check=True)
    subprocess.run([binary], check=True,
                   env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                        'UBSAN_OPTIONS': 'halt_on_error=1'})
