"""I build decoder fixtures through the same native and NanoISA routes."""
from pathlib import Path
import os
import shlex
import sys

from tests.native_process import native_link_flags, run_captured

ROOT = Path(__file__).resolve().parents[1]


def source_compiler_command():
    """I invoke the actual chosen native compiler or raw compiler module."""
    native = os.environ.get('NANOLANG_TRANSLATOR_SOURCE_COMPILER')
    module = os.environ.get('NANOLANG_TRANSLATOR_SOURCE_MODULE')
    if native and module:
        raise ValueError('I require one explicit NanoLang compiler product.')
    if module:
        path = Path(module)
        if not path.is_file():
            raise FileNotFoundError(path)
        return [ROOT / 'bin/nano_vm', path, '--']
    return [native] if native else []


def checked(command, *, build=False, timeout=60):
    completed = run_captured(command, build=build, timeout=timeout, cwd=ROOT)
    if completed.returncode:
        raise AssertionError(f'{command}: {completed.returncode}\n'
                             + completed.stdout.decode(errors='replace')
                             + completed.stderr.decode(errors='replace'))
    return completed.stdout.decode()


def build_routes(fixture, directory, *, legacy_native=True):
    native = directory / 'decoder-native'
    module = directory / 'decoder.nvm'
    source = directory / 'decoder.c'
    aot = directory / 'decoder-aot'
    if legacy_native:
        checked([ROOT / 'bin/nanoc_c', fixture, '-o', native], build=True)
    checked([ROOT / 'bin/nano_virt', fixture, '-o', module], build=True)
    compiler = shlex.split(os.environ.get('CC', 'cc'))

    def native_from_module(input_module, c_path, executable):
        checked([ROOT / 'bin/nvm2c', input_module, '-o', c_path], build=True)
        checked([*compiler, '-std=c11', '-O1', '-g',
                 '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                 c_path, ROOT / 'bin/nano_aot_runtime.o', '-lm',
                 *(['-ldl', '-Wl,--export-dynamic'] if sys.platform.startswith('linux') else []),
                 *native_link_flags(), '-o', executable], build=True)

    native_from_module(module, source, aot)
    routes = [('c-producer-nanoisa-vm', [ROOT / 'bin/nano_vm', module, '--']),
              ('c-producer-sanitized-native', [aot])]
    if legacy_native:
        routes.insert(0, ('c-seed-native', [native]))
    source_compiler = source_compiler_command()
    if source_compiler:
        source_module = directory / 'source-decoder.nvm'
        source_c = directory / 'source-decoder.c'
        source_native = directory / 'source-decoder-aot'
        checked([*source_compiler, fixture, '--emit-nvm', '-o', source_module], build=True)
        native_from_module(source_module, source_c, source_native)
        routes.extend([
            ('nanolang-producer-nanoisa-vm', [ROOT / 'bin/nano_vm', source_module, '--']),
            ('nanolang-producer-sanitized-native', [source_native])])
    return routes
