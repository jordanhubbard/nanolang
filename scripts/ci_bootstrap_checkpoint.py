#!/usr/bin/env python3
"""I transfer a successful instrumented bootstrap without changing its identity.

I retain legacy wire versions and use lossless version-three tables only when
the complete version-two manifest exceeds my unchanged 32 MiB bound. I check
composed link targets before packing, reading and restoring. I require a stable
checkout; these preflight checks do not isolate concurrent hostile mutations.
"""
import argparse
import base64
from collections import deque
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
    if stat.S_ISDIR(info.st_mode):
        return dict(kind='directory', mode=stat.S_IMODE(info.st_mode),
                    mtime_ns=info.st_mtime_ns)
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
        require(value['kind'] in ('file', 'link', 'directory'), 'known checkpoint entry kinds')
        for parent in PurePosixPath(name).parents:
            require(str(parent) not in records or records[str(parent)]['kind'] == 'directory',
                    'only directory members as archived parents')
    def target_parts(target):
        require(isinstance(target, str) and '\0' not in target and
                0 < len(os.fsencode(target)) <= 4095,
                'representable symbolic link targets')
        parts = PurePosixPath(target).parts
        absolute = PurePosixPath(target).is_absolute()
        if absolute:
            require(parts[:len(root.parts)] == root.parts,
                    'absolute links confined to the original checkout')
            parts = parts[len(root.parts):]
        return absolute, parts

    for name, value in records.items():
        if value['kind'] != 'link':
            continue
        absolute, parts = target_parts(value['target'])
        pending = deque(parts if absolute else PurePosixPath(name).parent.parts + parts)
        resolved, expansions, steps = [], 1, 0
        while pending:
            component = pending.popleft()
            steps += 1
            require(steps <= 4096, 'bounded symbolic link path traversal')
            if component == '.':
                continue
            if component == '..':
                require(bool(resolved), 'links confined to the original checkout')
                resolved.pop()
                continue
            candidate = '/'.join([*resolved, component])
            entry = records.get(candidate)
            existing = root / candidate
            target = None
            if entry is not None:
                if entry['kind'] == 'link':
                    target = entry['target']
                elif pending:
                    require(entry['kind'] == 'directory', 'directory link-target parents')
            elif existing.is_symlink():
                target = os.readlink(existing)
            elif pending:
                require(not existing.exists() or existing.is_dir(), 'directory link-target parents')
            if target is None:
                resolved.append(component)
                continue
            # I expand the archived graph before processing a later '..'.
            # Lexical normalization would erase exactly that distinction.
            expansions += 1
            require(expansions <= 40, 'bounded acyclic symbolic link resolution')
            absolute, parts = target_parts(target)
            if absolute:
                resolved.clear()
            pending.extendleft(reversed(parts))


MAX_MANIFEST_BYTES = 32 * 1024 * 1024


def visible_source_directories(root):
    """I retain the directory prerequisites used by bootstrap_input_tree."""
    names = set()

    def visit(path):
        if not path.is_dir():
            return
        require(not path.is_symlink(), 'ordinary bootstrap source directories')
        names.add(str(path.relative_to(root)))
        for child in path.iterdir():
            if not child.name.startswith('.'):
                visit(child)

    for name in ('src', 'src_nano', 'modules', 'std', 'stdlib'):
        visit(root / name)
    return names


def encode_manifest(manifest):
    # I bind source hashes before omitting their duplicate wire representation.
    wire = dict(manifest)
    if 'tracked' in wire:
        require(all(name in wire['files'] and wire['files'][name]['sha256'] == digest
                    for name, digest in wire['tracked'].items()),
                'tracked hashes matching their archived file records')
        positions = {name: index for index, name in enumerate(sorted(wire['files']))}
        wire['tracked'] = sorted(positions[name] for name in wire['tracked'])
    wire['version'] = 2
    encoded = json.dumps(wire, sort_keys=True, separators=(',', ':')).encode()
    if len(encoded) > MAX_MANIFEST_BYTES:
        del encoded
        # I intern repeated archive names/content identities, not omit records.
        wire['version'] = 3
        wire['files'] = compact_file_records(manifest['files'])
        encoded = json.dumps(wire, sort_keys=True, separators=(',', ':')).encode()
    require(len(encoded) <= MAX_MANIFEST_BYTES, 'a bounded regular checkpoint manifest')
    return encoded


def file_identity(value):
    kind = value.get('kind')
    fields = {'kind', 'mode', 'mtime_ns'}
    require(type(value.get('mode')) is int and 0 <= value['mode'] <= 0o7777 and
            type(value.get('mtime_ns')) is int, 'exact file modes and nanosecond times')
    if kind == 'directory':
        require(set(value) == fields, 'complete directory identity fields')
        return (2, value['mode'])
    require(kind in ('file', 'link'), 'known compact file kinds')
    require(isinstance(value.get('sha256'), str) and
            re.fullmatch('[0-9a-f]{64}', value['sha256']), 'exact compact file hashes')
    field = 'size' if kind == 'file' else 'target'
    require(set(value) == fields | {'sha256', field}, 'complete compact file identity fields')
    if kind == 'file':
        require(type(value['size']) is int and value['size'] >= 0, 'nonnegative regular file sizes')
    else:
        require(isinstance(value['target'], str) and '\0' not in value['target'], 'literal symbolic link targets')
    return (0 if kind == 'file' else 1, value['mode'], value[field], value['sha256'])


def compact_file_records(files):
    directories, basenames, identities = {}, set(), set()
    for name, value in files.items():
        safe_name(name)
        identity = file_identity(value)
        parent, _, basename = name.rpartition('/')
        directories[parent] = min(directories.get(parent, value['mtime_ns']), value['mtime_ns'])
        basenames.add(basename)
        identities.add(identity)
    parents, names, contents = sorted(directories), sorted(basenames), sorted(identities)
    parent_index = {name: index for index, name in enumerate(parents)}
    name_index = {name: index for index, name in enumerate(names)}
    content_index = {value: index for index, value in enumerate(contents)}
    entries = []
    for name in sorted(files):
        parent, _, basename = name.rpartition('/')
        value = files[name]
        entries.append([parent_index[parent], name_index[basename],
                        content_index[file_identity(value)], value['mtime_ns'] - directories[parent]])
    packed_contents = []
    for identity in contents:
        row = list(identity)
        if row[0] != 2:
            row[3] = base64.b85encode(bytes.fromhex(row[3])).decode('ascii')
        packed_contents.append(row)
    return {'directories': [[parent, directories[parent]] for parent in parents],
            'names': names, 'identities': packed_contents, 'entries': entries}


def expand_file_records(compact):
    require(isinstance(compact, dict) and set(compact) ==
            {'directories', 'names', 'identities', 'entries'}, 'the complete compact file tables')
    directories, names, identities, entries = (compact[key] for key in
                                               ('directories', 'names', 'identities', 'entries'))
    require(all(isinstance(table, list) for table in (directories, names, identities, entries)),
            'compact file table arrays')
    parents = []
    for row in directories:
        require(isinstance(row, list) and len(row) == 2 and isinstance(row[0], str) and
                '\0' not in row[0] and type(row[1]) is int, 'directory names and exact base times')
        if row[0]: safe_name(row[0])
        parents.append(row[0])
    require(parents == sorted(set(parents)), 'unique sorted compact directories')
    require(all(isinstance(name, str) and name and '/' not in name and '\0' not in name and
                name not in ('.', '..') for name in names), 'literal compact basenames')
    require(names == sorted(set(names)), 'unique sorted compact basenames')
    content = []
    for row in identities:
        require(isinstance(row, list) and len(row) in (2, 4) and type(row[0]) is int and
                row[0] in (0, 1, 2), 'known compact file identity rows')
        kind = row[0]
        require(len(row) == (2 if kind == 2 else 4), 'exact compact identity field counts')
        value = {'kind': ('file', 'link', 'directory')[kind], 'mode': row[1], 'mtime_ns': 0}
        if kind != 2:
            value['size' if kind == 0 else 'target'] = row[2]
            require(isinstance(row[3], str) and len(row[3]) == 40, 'fixed-width compact SHA256 values')
            try:
                digest = base64.b85decode(row[3])
            except (ValueError, TypeError):
                raise ValueError('I require a canonical compact SHA256 value') from None
            require(len(digest) == 32 and base64.b85encode(digest).decode('ascii') == row[3],
                    'canonical compact SHA256 values')
            value['sha256'] = digest.hex()
        file_identity(value)
        content.append(value)
    decoded_identities = [file_identity(value) for value in content]
    require(decoded_identities == sorted(set(decoded_identities)),
            'unique sorted compact content identities')
    result, order, used = {}, [], [set(), set(), set()]
    for row in entries:
        require(isinstance(row, list) and len(row) == 4 and all(type(value) is int for value in row),
                'four integer compact entry fields')
        directory, basename, identity, delta = row
        require(0 <= directory < len(directories) and 0 <= basename < len(names) and
                0 <= identity < len(content) and delta >= 0, 'bounded compact indices and time deltas')
        parent = parents[directory]
        name = (parent + '/' if parent else '') + names[basename]
        safe_name(name)
        require(name not in result, 'unique compact file paths')
        result[name] = {**content[identity], 'mtime_ns': directories[directory][1] + delta}
        order.append(name)
        for index, value in enumerate(row[:3]): used[index].add(value)
    require(order == sorted(order) and all(indices == set(range(len(table)))
            for indices, table in zip(used, (directories, names, identities))),
            'sorted paths and complete compact table references')
    return result


def decode_manifest(encoded):
    require(isinstance(encoded, (str, bytes, bytearray)), 'an encoded checkpoint manifest')
    require(len(encoded.encode() if isinstance(encoded, str) else encoded) <= MAX_MANIFEST_BYTES,
            'a bounded regular checkpoint manifest')
    manifest = json.loads(encoded)
    require(isinstance(manifest, dict), 'a checkpoint manifest object')
    require(manifest.get('version') in (1, 2, 3), 'checkpoint version one, two or three')
    if manifest['version'] == 3:
        manifest['files'] = expand_file_records(manifest.get('files'))
    if manifest['version'] in (2, 3) and 'tracked' in manifest:
        indices = manifest['tracked']
        names = sorted(manifest['files'])
        require(isinstance(indices, list) and all(type(i) is int for i in indices) and
                indices == sorted(set(indices)) and all(0 <= i < len(names) for i in indices),
                'unique sorted tracked indices with archived file records')
        require(all('sha256' in manifest['files'][names[i]] for i in indices),
                'tracked indices naming hashed file records')
        manifest['tracked'] = {names[i]: manifest['files'][names[i]]['sha256'] for i in indices}
    # I keep the in-memory identity contract unchanged for both wire versions.
    manifest['version'] = 1
    return manifest


def write_archive(root, names, metadata, destination):
    records = {name: record(root / name) for name in sorted(set(names))}
    validate_links(records, root)
    manifest = {**metadata, 'version': 1, 'files': records}
    encoded = encode_manifest(manifest)
    with tarfile.open(destination, 'w:gz', format=tarfile.PAX_FORMAT) as archive:
        member = tarfile.TarInfo('checkpoint.json'); member.size = len(encoded)
        archive.addfile(member, io.BytesIO(encoded))
        for name, value in records.items():
            member = tarfile.TarInfo('files/' + name)
            member.mode = value['mode']; member.mtime = value['mtime_ns'] / 1e9
            if value['kind'] == 'directory':
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            elif value['kind'] == 'link':
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
        require(manifest_member.isfile() and manifest_member.size <= MAX_MANIFEST_BYTES,
                'a bounded regular checkpoint manifest')
        manifest = decode_manifest(archive.extractfile(manifest_member).read())
        records = manifest['files']; validate_links(records, root)
        require(set(names) == {'checkpoint.json'} | {'files/' + name for name in records},
                'exact manifest coverage of archive members')
        for name, value in records.items():
            member = archive.getmember('files/' + name)
            require(member.mode == value['mode'], 'recorded file modes')
            if value['kind'] == 'directory':
                require(member.isdir() and member.size == 0, 'recorded directory identity')
            elif value['kind'] == 'link':
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
    # I repeat graph validation against the actual destination before any write.
    validate_links(manifest['files'], root)
    # I reject pre-existing escaping parent links before replacing any member.
    for name, value in manifest['files'].items():
        parent = (root / name).parent
        require(parent.resolve().is_relative_to(root), 'checkout-confined existing parents')
        target = root / name
        if value['kind'] == 'directory':
            require(not target.is_symlink() and
                    (not target.exists() or target.is_dir()),
                    'ordinary existing directory destinations')
    with tarfile.open(path, 'r:gz') as archive:
        for name, value in manifest['files'].items():
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
            if value['kind'] == 'directory':
                target.mkdir(exist_ok=True)
                continue
            if target.is_symlink(): target.unlink()
            if value['kind'] == 'link':
                if target.exists(): target.unlink()
                target.symlink_to(value['target'])
            else:
                with archive.extractfile('files/' + name) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
                target.chmod(value['mode'])
            os.utime(target, ns=(value['mtime_ns'], value['mtime_ns']), follow_symlinks=False)
    # Creating children changes their parents. I restore directory prerequisites
    # only after every member has been installed, never by inventing old dates.
    directories = ((name, value) for name, value in manifest['files'].items()
                   if value['kind'] == 'directory')
    for name, value in sorted(directories, key=lambda item: len(PurePosixPath(item[0]).parts), reverse=True):
        target = root / name
        target.chmod(value['mode'])
        os.utime(target, ns=(value['mtime_ns'], value['mtime_ns']), follow_symlinks=False)
    require(all(record(root / name) == value for name, value in manifest['files'].items()),
            'exact restored contents, modes, links and modification times')


def execution_image(identity_path=Path('/etc/nanolang-ci-container.json'), dockerenv=Path('/.dockerenv')):
    # The host wrapper checks the Docker config/platform before create and the
    # created container's actual Image before execution, then mounts this receipt
    # read-only. I do not accept an environment variable naming an unproved image.
    require(dockerenv.is_file() and identity_path.is_file(), 'the verified CI container execution receipt')
    value = json.loads(identity_path.read_text())
    require(set(value) == {'version', 'image_id', 'config_id', 'platform', 'archive_sha256'} and
            value['version'] == 1 and
            re.fullmatch(r'sha256:[0-9a-f]{64}', value['image_id']) and
            re.fullmatch(r'sha256:[0-9a-f]{64}', value['config_id']) and
            re.fullmatch(r'[0-9a-f]{64}', value['archive_sha256']) and
            value['platform'] in ('linux/amd64', 'linux/arm64'), 'a complete verified container identity')
    architecture = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(platform.machine())
    require(value['platform'] == 'linux/' + str(architecture), 'the actual container architecture')
    return value


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
                image=execution_image(),
                packages=sha(Path('/etc/nanolang-ci-packages').read_bytes()),
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
    require(all(name in manifest['files'] and manifest['files'][name]['sha256'] == digest
                for name, digest in sources.items()),
            'archive source bytes matching the unchanged tracked checkout')


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
            names.update(visible_source_directories(ROOT))
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
                        'unchanged flags, toolchain, runtime dependencies and verified container image')
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
