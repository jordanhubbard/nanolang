"""I retain explicit compiler commands and native runtime instrumentation in tests."""
import os
import platform
import shlex
import subprocess
from pathlib import Path


def native_cc():
    selected = (os.environ.get('NANO_NATIVE_TEST_CC') or
                os.environ.get('NANO_CC') or os.environ.get('CC'))
    if selected:
        compiler = shlex.split(selected)
    else:
        homebrew = Path('/opt/homebrew/opt/llvm/bin/clang')
        if platform.system() == 'Darwin' and homebrew.is_file():
            sdk = os.environ.get('SDKROOT', '').strip()
            if not sdk:
                result = subprocess.run(
                    ['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-path'],
                    capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                sdk = result.stdout.strip()
            if not Path(sdk).is_dir():
                raise RuntimeError(f'I cannot find the selected macOS SDK: {sdk}')
            compiler = [str(homebrew), '-isysroot', sdk]
        else:
            compiler = ['cc']
    if not compiler:
        raise ValueError('I require a nonempty native compiler command.')
    return compiler


def native_link_flags():
    return shlex.split(os.environ.get('NANO_ARTIFACT_LDFLAGS',
                                     os.environ.get('NANO_LDFLAGS',
                                                    os.environ.get('LDFLAGS', ''))))
