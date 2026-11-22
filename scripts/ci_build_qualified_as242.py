#!/usr/bin/env python3
"""I build an already-supported assembler for an isolated Clang compatibility axis."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ci_qualify_as246 import command

OUT = ROOT / '.test_output/source-capture-diagnostic/qualified-as242'
URL = 'https://sourceware.org/pub/binutils/releases/binutils-2.42.tar.xz'
SHA256 = 'f6e4d41fd5fc778b06b7891457b3620da5ecea1006c6a4a41ae998109f85a800'
ANNOUNCEMENT = 'https://lists.gnu.org/archive/html/info-gnu/2024-01/msg00016.html'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    record = {'url': URL, 'release_announcement': ANNOUNCEMENT,
              'expected_source_sha256': SHA256,
              'limits': ['I build only a private compatibility-axis tool. I install nothing '
                         'and change no default/user-guide compiler or GNU admission. '
                         'Temporary extracted sources/build objects receive normal cleanup; '
                         'the complete source archive, build logs and actual assembler remain.']}
    try:
        archive = OUT / 'binutils-2.42.tar.xz'
        with urllib.request.urlopen(URL, timeout=60) as response:
            with archive.open('wb') as target:
                shutil.copyfileobj(response, target)
        record['source_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        if record['source_sha256'] != SHA256:
            raise RuntimeError('I refuse a source archive that differs from its upstream release checksum.')
        with tempfile.TemporaryDirectory(prefix='nano-qualified-as242-') as temporary:
            root = Path(temporary)
            with tarfile.open(archive) as source:
                source.extractall(root, filter='data')
            build = root / 'build'
            build.mkdir()
            configure = [str(root / 'binutils-2.42/configure'), '--disable-nls',
                         '--disable-werror', '--disable-gdb', '--disable-gdbserver',
                         '--disable-gprofng', '--disable-gprof', '--disable-binutils',
                         '--disable-ld', '--disable-gold', 'CFLAGS=-O2 -g -std=gnu17']
            record['configure'] = configure
            command(configure, 'configure', 180, cwd=build, output=OUT)
            command(['make', '-j2', 'all-gas'], 'build', 600, cwd=build, output=OUT)
            (OUT / 'bin').mkdir()
            assembler = OUT / 'bin/as'
            shutil.copy2(build / 'gas/as-new', assembler)
            assembler.chmod(0o700)
            command([str(assembler), '--version'], 'assembler-version', 30,
                    cwd=ROOT, output=OUT)
            version = (OUT / 'assembler-version.stdout').read_text()
            if not version.splitlines()[0].endswith(') 2.42'):
                raise RuntimeError('I require the actual supported assembler version.')
            record['assembler_sha256'] = hashlib.sha256(assembler.read_bytes()).hexdigest()
            record['assembler_version'] = version
            record['compiler'] = str(Path(shutil.which('cc')).resolve())
            record['compiler_sha256'] = hashlib.sha256(Path(record['compiler']).read_bytes()).hexdigest()
            record['completed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        (OUT / 'build.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
