#!/usr/bin/env python3
"""I transfer a successful instrumented bootstrap without changing its identity."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ('CFLAGS', 'LDFLAGS', 'CC', 'NANO_CC', 'NANO_CFLAGS', 'NANO_LDFLAGS',
         'BOOTSTRAP_NANOISA_TIMEOUT', 'NANO_SHADOW_TIMEOUT_SECONDS', 'ASAN_OPTIONS', 'UBSAN_OPTIONS')
LABELS = ('seed', 'seed-verify', 'stage1', 'stage1-verify', 'stage1-translate',
          'stage1-native', 'stage1-hello-compile', 'stage1-hello-execute')


def require(condition, message):
    if not condition:
        raise ValueError('I require ' + message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def command(argv, root=ROOT):
    return subprocess.check_output(argv, cwd=root, text=True, timeout=120).strip()


def tracked(root):
    return subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')[:-1]


def safe_name(name):
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and '..' not in path.parts and
            '.' not in path.parts and '.git' not in path.parts and str(path) == name,
            'a canonical checkout-relative checkpoint path')


def record(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(path)
        return dict(kind='link', target=target, sha256=sha(os.fsencode(target)),
                    mode=stat.S_IMODE(info.st_mode), mtime_ns=info.st_mtime_ns)
    require(stat.S_ISREG(info.st_mode), 'regular files or explicit symbolic links')
    return dict(kind='file', sha256=sha(path.read_bytes()), size=info.st_size,
                mode=stat.S_IMODE(info.st_mode), mtime_ns=info.st_mtime_ns)


def validate_links(records, root):
    root = root.resolve()
    for name, value in records.items():
        safe_name(name)
        require(value['kind'] in ('file', 'link'), 'known checkpoint entry kinds')
        for parent in PurePosixPath(name).parents:
            require(str(parent) not in records or records[str(parent)]['kind'] != 'link',
                    'no checkpoint file beneath an archived symbolic link')
        if value['kind'] == 'link':
            target = Path(value['target'])
            if not target.is_absolute():
                target = root / name / '..' / target
            normalized = Path(os.path.abspath(target))
            require(normalized.is_relative_to(root), 'links confined to the original checkout')


def write_archive(root, names, metadata, destination):
    records = {name: record(root / name) for name in sorted(set(names))}
    validate_links(records, root)
    manifest = {**metadata, 'version': 1, 'files': records}
    encoded = json.dumps(manifest, sort_keys=True).encode()
    with tarfile.open(destination, 'w:gz', format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo('checkpoint.json'); member.size = len(encoded)
        archive.addfile(member, io.BytesIO(encoded))
        for name, value in records.items():
            member = tarfile.TarInfo('files/' + name)
            member.mode = value['mode']; member.mtime = value['mtime_ns'] / 1e9
            if value['kind'] == 'link':
                member.type = tarfile.SYMTYPE; member.linkname = value['target']
                archive.addfile(member)
            else:
                member.size = value['size']
                with (root / name).open('rb') as source:
                    archive.addfile(member, source)
    return manifest


def read_archive(path, root):
    # I validate all members and bytes before writing any checkout path.
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)) and names.count('checkpoint.json') == 1,
                'unique archive members and one manifest')
        manifest_member = archive.getmember('checkpoint.json')
        require(manifest_member.isfile() and manifest_member.size <= 32 * 1024 * 1024,
                'a bounded regular checkpoint manifest')
        manifest = json.load(archive.extractfile(manifest_member))
        require(manifest.get('version') == 1, 'checkpoint version one')
        records = manifest['files']; validate_links(records, root)
        require(set(names) == {'checkpoint.json'} | {'files/' + name for name in records},
                'exact manifest coverage of archive members')
        for name, value in records.items():
            member = archive.getmember('files/' + name)
            require(member.mode == value['mode'], 'recorded file modes')
            if value['kind'] == 'link':
                require(member.issym() and member.linkname == value['target'] and
                        sha(os.fsencode(member.linkname)) == value['sha256'], 'recorded link identity')
            else:
                require(member.isfile() and member.size == value['size'], 'recorded regular file sizes')
                h = hashlib.sha256()
                with archive.extractfile(member) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''): h.update(block)
                require(h.hexdigest() == value['sha256'], 'recorded provider bytes')
        return manifest


def restore_files(path, root, manifest):
    root = root.resolve()
    # I reject pre-existing escaping parent links before replacing any member.
    for name in manifest['files']:
        parent = (root / name).parent
        require(parent.resolve().is_relative_to(root), 'checkout-confined existing parents')
    with tarfile.open(path, 'r:gz') as archive:
        for name, value in manifest['files'].items():
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink(): target.unlink()
            if value['kind'] == 'link':
                if target.exists(): target.unlink()
                target.symlink_to(value['target'])
            else:
                with archive.extractfile('files/' + name) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
                target.chmod(value['mode'])
            os.utime(target, ns=(value['mtime_ns'], value['mtime_ns']), follow_symlinks=False)
    require(all(record(root / name) == value for name, value in manifest['files'].items()),
            'exact restored contents, modes, links and modification times')


def environment(receipt):
    tools = {}
    for name in ('cc', 'make', 'as', 'ld', sys.executable):
        path = Path(shutil.which(name) or name).resolve()
        require(path.is_file(), 'available checkpoint toolchain tools')
        tools[str(path)] = sha(path.read_bytes())
    for selector in ('-print-prog-name=cc1', '-print-file-name=libgcc.a'):
        path = Path(command(['cc', selector])).resolve()
        require(path.is_file(), 'the recorded C compiler support files')
        tools[str(path)] = sha(path.read_bytes())
    dependencies = {}
    providers = [ROOT / 'bin' / name for name in receipt['tools'] if not name.endswith('.o')]
    providers += [ROOT / 'bin/nanoc_stage1'] + [Path(name) for name in receipt['hosts']]
    for provider in providers:
        result = subprocess.run(['ldd', str(provider)], capture_output=True, text=True, timeout=30)
        require(result.returncode == 0, 'inspectable dynamic provider dependencies')
        require('not found' not in result.stdout, 'resolved dynamic dependencies')
        for line in result.stdout.splitlines():
            match = re.search(r'(?:=>\s+)?(/\S+)\s+\(', line)
            if match:
                path = Path(match[1]).resolve()
                if not path.is_relative_to(ROOT): dependencies[str(path)] = sha(path.read_bytes())
    return dict(flags={name: os.environ.get(name) for name in FLAGS}, tools=tools,
                dependencies=dependencies, machine=platform.machine(), python=sys.version,
                image={name: os.environ.get(name) for name in ('ImageOS', 'ImageVersion')},
                os_release=Path('/etc/os-release').read_text())


def require_stage1(receipt):
    require(receipt['phase'] == 'stage1' and tuple(s['label'] for s in receipt['steps']) == LABELS and
            all(s.get('exit_code') == 0 for s in receipt['steps']), 'a completely successful Stage 1 checkpoint')


FINAL_LABELS = LABELS + ('stage2', 'stage2-verify', 'stage2-translate', 'stage2-native',
                         'stage2-hello-compile', 'stage2-hello-execute',
                         'seed-final-verify', 'stage1-final-verify', 'stage2-final-verify')

def require_complete(receipt, phase):
    if phase == 'stage1':
        require_stage1(receipt)
    else:
        require(receipt['phase'] == 'stage2' and receipt.get('raw_stage1_stage2_equal') is True and
                tuple(s['label'] for s in receipt['steps']) == FINAL_LABELS and
                all(s.get('exit_code') == 0 for s in receipt['steps']),
                'both successful generations and all final verifications')

def require_identity(manifest, root, commit, sources):
    require(manifest['root'] == str(root) and manifest['commit'] == commit,
            'the original checkout root and exact commit')
    require(manifest['tracked'] == sources, 'the exact tracked source tree before restoring providers')


def require_continuation(current, prior, phase):
    require_complete(prior, phase)
    require_complete(current, 'stage2')
    require(current['work'] == prior['work'] and
            current['steps'][:len(prior['steps'])] == prior['steps'],
            'continuation of the original successful checkpoint, not a replacement build')
    require(all(current['artifacts'][name] == value for name, value in prior['artifacts'].items()),
            'unchanged checkpoint artifacts after continuation')


def checkpoint(phase="stage1"):
    from bootstrap_nanoisa import Bootstrap
    bootstrap = Bootstrap(ROOT, int(os.environ['BOOTSTRAP_NANOISA_TIMEOUT']))
    bootstrap.load()
    receipt = bootstrap.manifest
    require_complete(receipt, phase)
    bootstrap.check_artifacts(['nanoc_seed.nvm', 'nanoc_stage1.nvm', 'nanoc_stage1'])
    if phase == 'stage2':
        bootstrap.check_artifacts(['nanoc_stage2.nvm', 'nanoc_stage2'])
    for step in receipt['steps']:
        require(sha(Path(step['log']).read_bytes()) == step['log_sha256'], 'unchanged completed stage logs')
    return receipt


def metadata(phase):
    require(platform.system() == 'Linux', 'the Linux instrumented worker')
    receipt = checkpoint(phase)
    names = tracked(ROOT)
    require(subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=ROOT).returncode == 0,
            'unchanged tracked checkout sources')
    return dict(root=str(ROOT), commit=command(['git', 'rev-parse', 'HEAD']),
                tracked={name: record(ROOT / name)['sha256'] for name in names},
                environment=environment(receipt), receipt=receipt, phase=phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('pack', 'restore', 'verify-resume'))
    parser.add_argument('archive', type=Path)
    parser.add_argument('--phase', choices=('stage1', 'stage2'), default='stage1')
    args = parser.parse_args()
    try:
        if args.operation == 'pack':
            meta = metadata(args.phase); names = set(meta['tracked'])
            for directory in ('bin', 'obj', 'lib', 'modules'):
                names.update(str(p.relative_to(ROOT)) for p in (ROOT / directory).rglob('*')
                             if p.is_file() or p.is_symlink())
            names.update(('.bootstrap0.built', '.bootstrap1.built'))
            if args.phase == 'stage2': names.update(('.bootstrap2.built', '.bootstrap3.built'))
            write_archive(ROOT, names, meta, args.archive)
            read_archive(args.archive, ROOT)
            print('I retained the successful instrumented bootstrap checkpoint.')
        else:
            manifest = read_archive(args.archive, ROOT)
            require_identity(manifest, ROOT, command(['git', 'rev-parse', 'HEAD']),
                             {name: record(ROOT / name)['sha256'] for name in tracked(ROOT)})
            if args.operation == 'restore':
                # External dependencies are checked after validated provider restoration,
                # before executing any restored compiler or make recipe.
                restore_files(args.archive, ROOT, manifest)
                require(environment(manifest['receipt']) == manifest['environment'],
                        'unchanged flags, toolchain, runtime dependencies and runner image')
                require(checkpoint(manifest['phase']) == manifest['receipt'], 'the exact completed checkpoint identity')
                plan = command(['make', '-f', 'Makefile.gnu', '-n', 'bootstrap3',
                                'CFLAGS=' + os.environ['CFLAGS'], 'LDFLAGS=' + os.environ['LDFLAGS']])
                (ROOT / '.test_output').mkdir(exist_ok=True)
                (ROOT / '.test_output/sanitizer-checkpoint-resume-plan.log').write_text(plan + '\n')
                forbidden = r'bootstrap_nanoisa\.py\s+stage1\b' if manifest['phase'] == 'stage1' else r'bootstrap_nanoisa\.py\s+(?:stage1|stage2|verify)\b'
                require(not re.search(forbidden, plan),
                        'continuation without a timestamp-induced completed-generation rebuild')
            else:
                from bootstrap_nanoisa import Bootstrap
                bootstrap = Bootstrap(ROOT, int(os.environ['BOOTSTRAP_NANOISA_TIMEOUT'])); bootstrap.load()
                current = checkpoint('stage2'); prior = manifest['receipt']
                require_continuation(current, prior, manifest['phase'])
                bootstrap.check_artifacts(['nanoc_seed.nvm', 'nanoc_stage1.nvm', 'nanoc_stage2.nvm',
                                           'nanoc_stage1', 'nanoc_stage2'])
                print('I verified continuation from the exact transferred bootstrap checkpoint.')
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(str(error), file=sys.stderr); return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
