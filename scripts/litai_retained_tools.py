#!/usr/bin/env python3
"""I build the three Literate AI retained tool binaries and the host-toolchain pins they enforce.

Literate AI accepts only hashed native binaries as toolchain commands, and nanolang builds with the host's Xcode
toolchain, SDK, Python and Homebrew OpenSSL. I record the SHA-256 of each of those files in pins.txt and compile one
launcher that re-hashes them before it runs anything. This makes a changed host toolchain fail closed. It does not
make the toolchain hermetic: only the files named here are pinned, and system tools reached through PATH (perl, sed,
awk, grep, cp) are not. Darwin only. Run it, review the diff, then regenerate the retained manifest."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'literate/retained/tools'
SOURCE = TOOLS / 'src/retained_launcher.c'
NAMES = ('build-native', 'host-make', 'run-cases')
SYSTEM_PATH = '/usr/bin:/bin:/usr/sbin:/sbin'


def out(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover():
    if sys.platform != 'darwin':
        sys.exit('litai_retained_tools.py supports Darwin only; Linux needs its own launcher and pins')
    developer = out('xcode-select', '-p')
    path = ':'.join([f'{developer}/usr/bin', SYSTEM_PATH, '/opt/homebrew/bin'])
    env = {**os.environ, 'PATH': path}
    python = subprocess.run(['sh', '-c', 'command -v python3'], env=env, capture_output=True, text=True).stdout.strip()
    sdk = out('xcrun', '--show-sdk-path')
    files = {'/usr/bin/make', '/usr/bin/cc', '/usr/bin/clang', '/usr/bin/xcrun', '/bin/sh', '/bin/cp', '/bin/chmod', '/bin/rm', '/usr/bin/find', python,
             f'{sdk}/SDKSettings.json'}
    for tool in ('make', 'cc', 'clang', 'ld', 'ar', 'ranlib', 'python3'):
        found = subprocess.run(['xcrun', '-f', tool], capture_output=True, text=True).stdout.strip()
        if found:
            files.add(found)
    openssl = sorted(Path('/opt/homebrew/Cellar').glob('openssl@4/*/lib/libcrypto*.dylib'))
    if not openssl:
        sys.exit('no Homebrew openssl@4 libcrypto found; the Makefile links -lcrypto from there')
    files.update(str(p.resolve()) for p in openssl)
    return path, sorted(f for f in files if f and Path(f).is_file())


def main():
    path, files = discover()
    lines = [f'PATH\t{path}'] + [f'{sha(f)}\t{f}' for f in files]
    (TOOLS / 'pins.txt').write_text('\n'.join(lines) + '\n')
    launcher = TOOLS / 'retained-launcher'
    subprocess.run(['cc', '-Wall', '-Wextra', '-Werror', '-std=c99', '-O2', '-o', str(launcher), str(SOURCE)], check=True)
    for name in NAMES:
        shutil.copy2(launcher, TOOLS / name)
        (TOOLS / name).chmod(0o755)
    launcher.unlink()
    print(f'pinned {len(files)} host files; wrote {", ".join(NAMES)} and pins.txt')
    for f in files:
        print(' ', f)


if __name__ == '__main__':
    main()
