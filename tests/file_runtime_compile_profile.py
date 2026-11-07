"""I share my actual File plan/private-VM recipes without changing native assertions."""
from pathlib import Path

HOSTED_PROVIDERS = [
    'file_flow', 'service_file_nominal_plan', 'service_bindings_module',
    'nvm_format', 'nvm_v2_constants', 'nvm_v2_signatures', 'nvm_v2_layouts',
    'nvm_v2_functions', 'nvm_v2_imports', 'nvm_v2_module',
    'nvm_v2_convert', 'retained_layouts', 'ownership_contracts',
]
VM_PROVIDERS = [*(f'src/nanoisa/{name}.c' for name in HOSTED_PROVIDERS),
                'src/nsi_cap.c', 'src/nsi_file.c', 'src/nsi_file_values.c',
                'src/nanoisa/file_runtime.c', 'src/nanovm/file_vm_private.c',
                'src/nanovm/vm_ffi.c']


def compile_flags(kind, extra=(), sanitizers=False):
    if kind == 'hosted':
        flags = [*extra, '-std=c11', '-D_DEFAULT_SOURCE', '-g', '-O1',
                 '-Wall', '-Wextra', '-Werror']
    elif kind == 'vm':
        flags = [*extra, '-std=c11', '-D_DEFAULT_SOURCE', '-DNVM_FILE_VM_PRIVATE',
                 '-g', '-O1', '-Wall', '-Wextra', '-Werror', '-Isrc', '-Isrc/nanoisa']
    else:
        raise ValueError('I require my hosted or private-VM File recipe.')
    if sanitizers:
        flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    return flags


def provider_profiles(kind, instrument):
    """I retain each original object source, actual input and selected hook vector."""
    if kind not in ('hosted', 'vm'):
        raise ValueError('I require my hosted or private-VM File recipe.')
    providers = [f'src/nanoisa/{name}.c' for name in HOSTED_PROVIDERS] if kind == 'hosted' else VM_PROVIDERS
    allocation = ['-include', 'tests/nanoisa/file_hosted_alloc.h' if kind == 'hosted'
                  else 'tests/nanoisa/file_runtime_hooks.h',
                  '-Dmalloc=file_test_malloc', '-Dcalloc=file_test_calloc',
                  '-Drealloc=file_test_realloc', '-Dfree=file_test_free']
    profiles = []
    for source in providers:
        stem = Path(source).stem
        if kind == 'vm' and instrument and stem in ('file_runtime', 'file_vm_private'):
            # I retain the fixture's exact-source inclusion and private accounting.
            continue
        hooks = allocation.copy() if instrument and (kind == 'hosted' or stem != 'vm_ffi') else []
        if kind == 'vm' and stem == 'nsi_file' and instrument:
            hooks += ['-DFILE_RUNTIME_HOST_HOOKS']
        if kind == 'vm' and stem == 'vm_ffi':
            hooks = ['-Dffi_loader_init=file_runtime_loader_init',
                     '-Dffi_loader_open=file_runtime_loader_open', '-Dfork=file_runtime_fork']
        actual = ('tests/nanoisa/file_runtime_frame_values.c'
                  if kind == 'vm' and instrument and stem == 'nsi_file_values' else source)
        profiles.append((source, actual, hooks))
    return profiles


def fixture_profile(kind, instrument):
    if kind not in ('hosted', 'vm'):
        raise ValueError('I require my hosted or private-VM File recipe.')
    source = 'tests/nanoisa/test_file_hosted.c' if kind == 'hosted' else 'tests/nanoisa/test_file_private_vm.c'
    return source, ['-DHOSTED_INSTRUMENT'] if instrument else []
