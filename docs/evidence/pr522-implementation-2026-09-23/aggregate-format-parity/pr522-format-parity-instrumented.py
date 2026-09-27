import os, shlex, subprocess, tempfile, unittest
from pathlib import Path
root = Path.cwd()
cc = '/opt/homebrew/opt/llvm/bin/clang'
flags = '-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer'
sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
env = os.environ.copy()
env.update(ASAN_OPTIONS='detect_leaks=1:detect_stack_use_after_return=1', UBSAN_OPTIONS='halt_on_error=1')
with tempfile.TemporaryDirectory(prefix='pr522-format-seed-') as directory:
    work = Path(directory)
    for name in ('src', 'modules', 'stdlib', 'src_nano'):
        (work / name).symlink_to(root / name, target_is_directory=True)
    cflags = '-Wall -Wextra -Werror -std=c99 -g -O1 -fPIC -Isrc -D_GNU_SOURCE ' + flags
    cflags += ' -I' + shlex.quote(sdk + '/usr/include/ffi') + ' -I/opt/homebrew/opt/openssl@3/include'
    command = ['make', '-j2', f'OBJ_DIR={work}/obj', f'BIN_DIR={work}/bin', f'CC={cc}',
               f'CFLAGS={cflags}', f'LDFLAGS=-lm -lcrypto -lffi -L/opt/homebrew/opt/openssl@3/lib {flags}',
               f'FILE_PUBLIC_LIBRARY={work}/lib/libnano_file_runtime.a', str(work / 'bin/nanoc_c')]
    subprocess.run(command, env=env, check=True)
    for name in ('eval', 'typechecker'):
        symbols = subprocess.check_output(['nm', '-u', str(work / 'obj' / (name + '.o'))], text=True)
        assert '__asan_' in symbols and '__ubsan_' in symbols, name
    print('I verified fresh ASan/UBSan eval and typechecker objects.', flush=True)
    os.environ.update(env)
    os.environ.update(CC=cc, NANO_CC=cc, NANO_CFLAGS='-O1 -g ' + flags, NANO_LDFLAGS=flags)
    import sys
    sys.path.insert(0, str(root))
    from tests import test_cseed_aggregate_formatting as tests
    original = subprocess.run
    def run(command, *args, **kwargs):
        command = list(command)
        if str(command[0]) == str(root / 'bin/nanoc_c'):
            command[0] = str(work / 'bin/nanoc_c')
        return original(command, *args, **kwargs)
    subprocess.run = run
    try:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    finally:
        subprocess.run = original
    raise SystemExit(not result.wasSuccessful())
