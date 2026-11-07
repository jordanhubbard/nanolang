"""I share the actual binding-harness compiler configuration with diagnostic planning."""
from pathlib import Path

PROVIDERS = ('src/nsi_file_binding.c', 'src/nsi_file_plan.c', 'src/nsi.c', 'src/cJSON.c', 'src/utf8.c')
KINDS = ('file', 'socket', 'websocket')


def provider_sources(kind):
    if kind not in KINDS:
        raise ValueError('I require a declared binding catalog kind.')
    return tuple(path.replace('file', kind) for path in PROVIDERS)


def compile_flags(root, work, extra=(), sanitizers=False):
    root, work = Path(root), Path(work)
    flags = [*extra, '-std=c99', '-D_GNU_SOURCE', '-D_DARWIN_C_SOURCE',
             '-Wall', '-Wextra', '-Werror', '-g', '-O1',
             '-I', str(root / 'src'), '-I', str(root / 'tests'), '-I', str(work)]
    if sanitizers:
        flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    return flags


def provider_hooks(root, mode):
    if mode not in ('linked', 'instrumented'):
        raise ValueError('I require a linked or instrumented binding mode.')
    return ['-include', str(Path(root) / 'tests/file_binding_hooks.h')] if mode == 'instrumented' else []


def fixture_hooks(mode):
    if mode not in ('linked', 'instrumented'):
        raise ValueError('I require a linked or instrumented binding mode.')
    return ['-DBINDING_INSTRUMENT'] if mode == 'instrumented' else []
