#!/usr/bin/env python3
"""I provision my pinned Wasm test runtime without a mutable install script."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

VERSION = "50.0.1"
# I retain the release asset digests from bytecodealliance/wasmtime v50.0.1.
ARCHIVES = {
    ("Linux", "x86_64"): "685466455dd42a14774a86c36c8f2e45cbb9bd6d92a850139315df60a75498a5",
    ("Linux", "aarch64"): "d737e077416d78fbfb0a008229ec27d6a0fd5a4924874b8c38b293c850cf4094",
    ("Darwin", "x86_64"): "ea1818fa8cbb51524847ab2ad22197e0051594fdc96cbd0cc2afb65832e11925",
    ("Darwin", "aarch64"): "960e5216c7ebd7e9b45e6561d15821b8b801b7d54aee28c7b91921aeb30c65c8",
}


def check(executable=None):
    executable = executable or shutil.which("wasmtime")
    if not executable:
        raise ValueError(f"I require wasmtime {VERSION}; run make install-test-wasmtime and add .test_output/test-tools/bin to PATH.")
    result = subprocess.run([str(executable), "--version"], capture_output=True,
                            text=True, timeout=15, check=True)
    if result.stdout.split()[:2] != ["wasmtime", VERSION]:
        raise ValueError(f"I require wasmtime {VERSION}; I found {result.stdout.strip()!r}.")


def install(archive, expected, member_name, prefix):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("I require the pinned Wasmtime archive SHA-256.")
    destination = prefix / "bin"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as source:
        member = source.getmember(member_name)
        if not member.isfile() or not 0 < member.size <= 256 * 1024 * 1024:
            raise ValueError("I require one regular Wasmtime executable below 256 MiB.")
        # I read only this member; archive paths and links are never extracted.
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as output:
            temporary = Path(output.name)
            try:
                with source.extractfile(member) as binary:
                    shutil.copyfileobj(binary, output)
                output.close()
                temporary.chmod(0o755)
                check(temporary)
                os.replace(temporary, destination / "wasmtime")
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if args.prefix is None:
        parser.error("I require an explicit --prefix or --check.")
    system = platform.system()
    machine = {"arm64": "aarch64", "AMD64": "x86_64"}.get(platform.machine(), platform.machine())
    expected = ARCHIVES.get((system, machine))
    if expected is None:
        raise ValueError(f"I have no pinned Wasmtime archive for {system}/{machine}.")
    name = f"wasmtime-v{VERSION}-{machine}-{'macos' if system == 'Darwin' else 'linux'}"
    url = f"https://github.com/bytecodealliance/wasmtime/releases/download/v{VERSION}/{name}.tar.xz"
    with tempfile.TemporaryDirectory(prefix="nano-wasmtime-download-") as temporary:
        archive = Path(temporary) / "runtime.tar.xz"
        with urllib.request.urlopen(url, timeout=120) as remote, archive.open("wb") as output:
            total = 0
            while chunk := remote.read(1024 * 1024):
                total += len(chunk)
                if total > 256 * 1024 * 1024:
                    raise ValueError("I require a Wasmtime archive below 256 MiB.")
                output.write(chunk)
        install(archive, expected, name + "/wasmtime", args.prefix)
    print(f"I installed wasmtime {VERSION} in {args.prefix / 'bin'}.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, tarfile.TarError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
