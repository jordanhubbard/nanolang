#!/usr/bin/env python3
"""I build explicit private assembler compatibility and source-repair profiles."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ci_qualify_as246 import command

OUT = ROOT / '.test_output/source-capture-diagnostic/qualified-as242'
URL = 'https://sourceware.org/pub/binutils/releases/binutils-2.42.tar.xz'
SHA256 = 'f6e4d41fd5fc778b06b7891457b3620da5ecea1006c6a4a41ae998109f85a800'
FIXED_SHA256 = 'd75a94f4d73e7a4086f7513e67e439e8fcdcbb726ffe63f4661744e6256b2cf2'
BEFORE = '40e7e14b5845e3f48b3be1cde0ea0b073f3e787864bdeb9d06dea9b545acfa07'
AFTER = '6ffe0ed82d02c8f92198b4c2b5200a8093b7eea0a07a96bf951eecabf021b036'
VENDOR = 'GNU Binutils for NanoLang retained capture r1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_include_fix(path):
    if digest(path) != BEFORE:
        raise RuntimeError('I refuse an unexpected upstream include-state source.')
    raw = path.read_text()
    old = 'if (logical_input_file == 0 && logical_input_line == -1u\n'
    new = ('if (from_sb_expansion != expanding_none\n'
           '\t  && logical_input_file == 0 && logical_input_line == -1u\n')
    if raw.count(old) != 1:
        raise RuntimeError('I require one exact include-state repair site.')
    candidate = raw.replace(old, new)
    if hashlib.sha256(candidate.encode()).hexdigest() != AFTER:
        raise RuntimeError('I refuse an unexpected upstream candidate delta.')
    path.write_text(candidate)


def test_environment(out):
    packages = out / 'packages'
    packages.mkdir()
    command(['apt-cache', 'depends', 'tcl-expect'], 'expect-dependencies', 30, output=out)
    dependency_text = (out / 'expect-dependencies.stdout').read_text()
    tcl = re.findall(r'^\s*Depends:\s+(libtcl[\w.]+)\s*$', dependency_text, re.MULTILINE)
    names = ['expect', 'tcl-expect', 'dejagnu', *tcl]
    command(['apt-get', 'download', *dict.fromkeys(names)], 'test-packages-download', 180,
            cwd=packages, output=out)
    private = out / 'private'
    records = {}
    for package in sorted(packages.glob('*.deb')):
        name = command(['dpkg-deb', '-f', str(package), 'Package'],
                       package.name + '-name', 30, output=out).stdout.decode().strip()
        version = command(['dpkg-deb', '-f', str(package), 'Version'],
                          package.name + '-version', 30, output=out).stdout.decode().strip()
        metadata = command(['apt-cache', 'show', name + '=' + version],
                           package.name + '-metadata', 30, output=out).stdout.decode()
        hashes = re.findall(r'^SHA256: (\w+)$', metadata, re.MULTILINE)
        if digest(package) not in hashes:
            raise RuntimeError('I require the private test package to match APT metadata.')
        records[name] = {'version': version, 'sha256': digest(package)}
        command(['dpkg-deb', '--extract', str(package), str(private)],
                package.name + '-extract', 30, output=out)
    libraries = {str(p.parent) for p in private.rglob('libexpect.so.*')}
    libraries.update(str(p.parent) for p in private.rglob('libtcl*.so*'))
    if not libraries:
        raise RuntimeError('I require the private native Expect library.')
    env = dict(os.environ, PATH=str(private / 'usr/bin') + os.pathsep + os.environ['PATH'],
               LD_LIBRARY_PATH=os.pathsep.join(sorted(libraries)),
               DEJAGNULIBS=str(private / 'usr/share/dejagnu'))
    env.pop('DEJAGNU', None)
    tcl_library = private / 'usr/share/tcltk/tcl8.6'
    if tcl_library.is_dir():
        env['TCL_LIBRARY'] = str(tcl_library)
    command([str(private / 'usr/bin/runtest'), '--version'], 'runtest-version', 30,
            env=env, output=out)
    return env, records


def gas_suite(build, out, label, env):
    clang = os.environ.get('NANO_AS_QUALIFICATION_CLANG_FOR_TARGET', shutil.which('clang'))
    if not clang or any(c.isspace() for c in clang):
        raise RuntimeError('I require an explicit native auxiliary Clang path.')
    command([clang, '-print-file-name=LLVMgold.so'], label + '-clang-query', 30, output=out)
    if (out / (label + '-clang-query.stderr')).read_bytes():
        raise RuntimeError('I require a clean auxiliary query before upstream test loading.')
    summary = build / 'gas/testsuite/gas.sum'
    log = build / 'gas/testsuite/gas.log'
    try:
        command(['make', '-j1', 'check-gas', 'RUNTESTFLAGS=CLANG_FOR_TARGET=' + clang],
                label + '-gas', 1200, cwd=build, env=env, output=out)
    finally:
        for file in (summary, log):
            if file.is_file():
                shutil.copy2(file, out / (label + '-' + file.name))
    rows = [line for line in summary.read_text().splitlines()
            if re.match(r'^(PASS|FAIL|XFAIL|XPASS|KFAIL|KPASS|UNSUPPORTED|UNRESOLVED|ERROR):', line)]
    if sum(line.startswith('PASS:') for line in rows) < 1000 or any(
            line.startswith(('FAIL:', 'XPASS:', 'UNRESOLVED:', 'ERROR:')) for line in rows):
        raise RuntimeError('I require the complete native GAS suite without unexpected outcomes.')
    return rows


def witness(before, after, readelf, out):
    source = ROOT / 'tests/fixtures/as246-source-identity'
    manifest = json.loads((source / 'manifest.json').read_text())
    target = out / 'witness'
    target.mkdir()
    for name, entry in manifest['files'].items():
        if digest(source / name) != entry['sha256']:
            raise RuntimeError('I refuse a changed retained native witness.')
        shutil.copy2(source / name, target / name)
    results = []
    for name in ('actual-markers.s', 'without-directory.s', 'without-predef.s', 'source-marker-only.s'):
        for label, tool in (('baseline', before), ('fixed', after)):
            obj = target / (label + '-' + Path(name).stem + '.o')
            command([str(tool), '--gdwarf-5', '-o', str(obj), str(target / name)],
                    label + '-' + Path(name).stem, 30, cwd=target, output=target)
            values = {}
            for key, option in (('lines', '--debug-dump=decodedline'), ('info', '--debug-dump=info')):
                values[key] = command([str(readelf), option, str(obj)],
                    label + '-' + Path(name).stem + '-' + key, 30, output=target).stdout.decode()
            actual = ('payload.S' in values['lines'] and any('DW_AT_name' in line and
                'payload.S' in line for line in values['info'].splitlines()))
            expected = not (label == 'baseline' and name == 'actual-markers.s')
            results.append({'case': name, 'tool': label, 'original_source_identity': actual,
                            'expected': expected, 'object_sha256': digest(obj)})
            (target / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            if actual != expected:
                raise RuntimeError('I require the original bug and all corrected neighbor decisions.')
    return results


def main(variant='as242'):
    if variant not in ('as242', 'fixed-as246'):
        raise ValueError('I require an explicit supported assembler profile.')
    fixed = variant == 'fixed-as246'
    out = ROOT / '.test_output/source-capture-diagnostic/fixed-as246' if fixed else OUT
    out.mkdir(parents=True, exist_ok=False)
    version = '2.46.0' if fixed else '2.42'
    url = 'https://sourceware.org/pub/binutils/releases/binutils-' + version + '.tar.xz'
    expected = FIXED_SHA256 if fixed else SHA256
    record = {'profile': variant, 'url': url, 'expected_source_sha256': expected,
              'limits': ['I build only a private identified tool. I install nothing and '
                         'change no default compiler or generic GNU admission. I retain source '
                         'archives, selected native products and logs; temporary build objects '
                         'receive normal cleanup. A tagged baseline image is an explicitly '
                         'unqualified negative control and is never selected for capture.']}
    def save(stage):
        record['stage'] = stage
        temporary = out / 'build.json.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(out / 'build.json')
    try:
        record['available_workspace_bytes'] = shutil.disk_usage(out).free
        record['available_temporary_bytes'] = shutil.disk_usage(tempfile.gettempdir()).free
        if min(record['available_workspace_bytes'], record['available_temporary_bytes']) < 5 * 1024**3:
            raise RuntimeError('I require five GiB of source/build headroom for this private tool.')
        save('download')
        archive = out / ('binutils-' + version + '.tar.xz')
        started = time.monotonic()
        with urllib.request.urlopen(url, timeout=60) as response:
            with archive.open('wb') as target:
                count = 0
                while True:
                    if time.monotonic() - started > 600:
                        raise TimeoutError('I retain a bounded source-download timeout.')
                    data = response.read(1024 * 1024)
                    if not data:
                        break
                    count += len(data)
                    if count > 64 * 1024 * 1024:
                        raise RuntimeError('I refuse an oversized source archive.')
                    target.write(data)
        record['download_seconds'] = time.monotonic() - started
        record['source_sha256'] = digest(archive)
        if record['source_sha256'] != expected:
            raise RuntimeError('I refuse a source archive that differs from its upstream release checksum.')
        with tempfile.TemporaryDirectory(prefix='nano-capture-assembler-') as temporary:
            root = Path(temporary)
            with tarfile.open(archive) as source:
                source.extractall(root, filter='data')
            source = root / ('binutils-' + version)
            build = root / 'build'
            build.mkdir()
            configure = [str(source / 'configure'), '--disable-nls', '--disable-werror',
                '--disable-gdb', '--disable-gdbserver', '--disable-gprofng', '--disable-gprof',
                '--disable-ld', '--disable-gold', 'CFLAGS=-O2 -g -std=gnu17']
            configure += ['--with-pkgversion=' + VENDOR] if fixed else ['--disable-binutils']
            record['configure'] = configure
            save('configure')
            command(configure, 'configure', 180, cwd=build, output=out)
            targets = ['all-gas', 'all-binutils'] if fixed else ['all-gas']
            save('build')
            command(['make', '-j2', *targets], 'build', 600, cwd=build, output=out)
            if fixed:
                path = source / 'gas/input-scrub.c'
                if digest(path) != BEFORE:
                    raise RuntimeError('I require the exact original upstream source.')
                shutil.copy2(path, out / 'input-scrub-before.c')
                before = out / 'as-before'
                shutil.copy2(build / 'gas/as-new', before)
                env, record['test_packages'] = test_environment(out)
                save('baseline upstream tests')
                baseline = gas_suite(build, out, 'baseline', env)
                apply_include_fix(path)
                shutil.copy2(path, out / 'input-scrub-fixed.c')
                record['fixed_source_sha256'] = digest(path)
                save('fixed source build')
                command(['make', '-j2', 'all-gas'], 'fixed-build', 600, cwd=build, output=out)
                after = out / 'as-after'
                shutil.copy2(build / 'gas/as-new', after)
                save('fixed upstream tests')
                corrected = gas_suite(build, out, 'fixed', env)
                if baseline != corrected:
                    raise RuntimeError('I require unchanged complete upstream test outcomes.')
                record['gas_cases'] = len(corrected)
                record['gas_expected_passes'] = sum(line.startswith('PASS:') for line in corrected)
                record['gas_unsupported'] = sum(line.startswith('UNSUPPORTED:') for line in corrected)
                for name in ('readelf', 'objdump'):
                    shutil.copy2(build / ('binutils/' + name), out / name)
                save('source identity witnesses')
                record['witnesses'] = witness(before, after, out / 'readelf', out)
            (out / 'bin').mkdir()
            assembler = out / 'bin/as'
            shutil.copy2(build / 'gas/as-new', assembler)
            assembler.chmod(0o700)
            command([str(assembler), '--version'], 'assembler-version', 30,
                    cwd=ROOT, output=out)
            actual = (out / 'assembler-version.stdout').read_text()
            if fixed:
                if VENDOR not in actual.splitlines()[0] or not actual.splitlines()[0].endswith(' 2.46.0.20260210'):
                    raise RuntimeError('I require the actual separately identified fixed assembler.')
            elif not actual.splitlines()[0].endswith(') 2.42'):
                raise RuntimeError('I require the actual supported assembler version.')
            record['assembler_sha256'] = digest(assembler)
            record['assembler_version'] = actual
            record['compiler'] = str(Path(shutil.which('cc')).resolve())
            record['compiler_sha256'] = digest(record['compiler'])
            record['completed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        save('complete' if record.get('completed') else 'refused')


if __name__ == '__main__':
    if sys.argv[1:] not in ([], ['--fixed-as246']):
        raise SystemExit('I require an explicit supported assembler profile.')
    main('fixed-as246' if sys.argv[1:] else 'as242')
