#!/usr/bin/env python3
"""I install and verify one complete, byte-identified native SDK generation."""
import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile

ABI = 2
LIMIT_FILES = 8192
LIMIT_BYTES = 1024 * 1024 * 1024
PUBLIC = ('nanoc', 'nanoc_c', 'nano_virt', 'nano_vm', 'nano_cop', 'nano_vmd', 'nanoisa', 'nvm2c')
HEADER = b'NANOLANG_SDK_ABI=2\n'
MANIFEST = 'sdk.inputs'


def relative(value):
    p = Path(value)
    if (not value or value.startswith('/') or '\\' in value or
            any(ord(c) < 32 or ord(c) == 127 for c in value) or
            any(x in ('', '.', '..') for x in value.split('/')) or
            len(value.encode()) > 2048):
        raise ValueError('I require a bounded relative SDK input path')
    return p


def owned_path(root, value):
    p = root
    for part in relative(value).parts:
        p = p / part
        if p.is_symlink():
            raise ValueError('I refuse a symlink in an SDK-owned path: ' + str(p))
    return p


def digest(path, expected_size):
    h = hashlib.sha256()
    amount = 0
    with path.open('rb') as stream:
        while True:
            data = stream.read(min(1024 * 1024, expected_size - amount + 1))
            if not data:
                break
            amount += len(data)
            if amount > expected_size:
                raise ValueError('I refuse an SDK input that grew during hashing')
            h.update(data)
    if amount != expected_size:
        raise ValueError('I refuse an SDK input that shrank during hashing')
    return h.hexdigest()


def row(path, source):
    st = source.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_size < 0 or st.st_size > LIMIT_BYTES:
        raise ValueError('I require a regular SDK input: ' + str(source))
    mode = stat.S_IMODE(st.st_mode)
    if mode & ~0o777:
        raise ValueError('I refuse special SDK input mode bits')
    return dict(path=path, size=st.st_size, mode=mode, sha256=digest(source, st.st_size))


def body(rows):
    return HEADER + b''.join(('%04o\t%d\t%s\t%s\n' %
                             (r['mode'], r['size'], r['sha256'], r['path'])).encode()
                            for r in rows)


def manifest(rows):
    data = body(rows)
    return hashlib.sha256(data).hexdigest(), data


def read_manifest(root):
    p = owned_path(root, MANIFEST)
    info = p.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024 * 1024:
        raise ValueError('I require a bounded regular SDK manifest')
    data = p.read_bytes()
    if not data.startswith(HEADER):
        raise ValueError('I require native array ABI2 in my SDK')
    rows = []
    total = 0
    for line in data[len(HEADER):].splitlines():
        mode, size, sha, name = line.decode('utf-8').split('\t')
        relative(name)
        if (len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) or
                mode != '%04o' % int(mode, 8) or size != str(int(size))):
            raise ValueError('I refuse a noncanonical SDK record')
        r = dict(path=name, size=int(size), mode=int(mode, 8), sha256=sha)
        if r['size'] < 0 or r['mode'] & ~0o777:
            raise ValueError('I refuse an invalid SDK extent or mode')
        total += r['size']
        rows.append(r)
    names = [r['path'] for r in rows]
    if (not rows or len(rows) > LIMIT_FILES or total > LIMIT_BYTES or
            names != sorted(set(names)) or MANIFEST in names or body(rows) != data):
        raise ValueError('I refuse an unordered or excessive SDK inventory')
    return hashlib.sha256(data).hexdigest(), rows


def verify(root, identity=None):
    root_state = root.lstat()
    if not stat.S_ISDIR(root_state.st_mode):
        raise ValueError('I require a real SDK generation directory')
    actual, rows = read_manifest(root)
    if identity is not None and actual != identity:
        raise ValueError('I refuse a mismatched SDK generation identity')
    expected = {MANIFEST}
    directories = set()
    for r in rows:
        p = owned_path(root, r['path'])
        if row(r['path'], p) != r:
            raise ValueError('I refuse a changed SDK input: ' + r['path'])
        expected.add(r['path'])
        directories.update(str(v) for v in Path(r['path']).parents if str(v) != '.')
    seen_files, seen_dirs = set(), set()
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(parent) / name
            rel = str(p.relative_to(root))
            mode = p.lstat().st_mode
            if stat.S_ISDIR(mode):
                seen_dirs.add(rel)
            elif stat.S_ISREG(mode):
                seen_files.add(rel)
            else:
                raise ValueError('I refuse an unexpected SDK entry: ' + rel)
    current = root.lstat()
    if (current.st_dev, current.st_ino) != (root_state.st_dev, root_state.st_ino):
        raise ValueError('I refuse a replaced SDK generation directory')
    if seen_files != expected or seen_dirs != directories:
        raise ValueError('I refuse unowned files or directories in an SDK generation')
    return actual, rows


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def rename_exclusive(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    if platform.system() == 'Linux':
        call = libc.renameat2
        call.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = call(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    elif platform.system() == 'Darwin':
        call = libc.renamex_np
        call.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = call(os.fsencode(source), os.fsencode(target), 4)
    else:
        raise ValueError('I require an authoritative exclusive directory rename')
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def ensure_dirs(path):
    if not path.is_absolute():
        raise ValueError('I require an absolute installation prefix')
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            current.mkdir()
            current.chmod(0o755)
        except FileExistsError:
            if not stat.S_ISDIR(current.lstat().st_mode):
                raise ValueError('I refuse a non-directory installation ancestor')


def install(source, prefix, inventory):
    source = source.resolve(strict=True)
    prefix = prefix.resolve()
    if source == prefix or source in prefix.parents or prefix in source.parents:
        raise ValueError('I refuse overlapping source and installation roots')
    names = json.loads(inventory.read_text())
    if not isinstance(names, list) or names != sorted(set(names)):
        raise ValueError('I require an ordered unique committed SDK inventory')
    selected = {name: owned_path(source, name) for name in names}
    objects = json.loads((source / 'scripts/native_sdk_objects.json').read_text())
    wrapper = (source / 'src/nanovirt/wrapper_gen.c').read_text()
    start = wrapper.index('static bool build_obj_list(')
    end = wrapper.index('    const char **groups[]', start)
    actual_objects = sorted(set('obj/' + item for item in re.findall(r'"([^"\n]+\.o)"', wrapper[start:end])))
    if objects != actual_objects or not objects:
        raise ValueError('I require the complete current wrapper object inventory')
    for name in objects:
        selected[name] = owned_path(source, name)
    for output, binary in [('nanoc_c', 'nanoc_c'), ('nanoc_stage1', 'nanoc_stage1'),
                           ('nanoc', 'nanoc_stage2')]:
        p = owned_path(source, 'bin/' + binary)
        result = subprocess.run([str(p), '--native-array-abi'], check=True,
                                capture_output=True, timeout=30)
        if result.stdout != b'2\n':
            raise ValueError('I require a freshly built ABI2 compiler before installation')
        selected['bin/' + output] = p
    for name in PUBLIC[2:]:
        selected['bin/' + name] = owned_path(source, 'bin/' + name)
    # The assembler helper is an actual Linux module-build input.
    if platform.system() == 'Linux':
        selected['bin/nano_as_capture.so'] = owned_path(source, 'bin/nano_as_capture.so')
    if len(selected) > LIMIT_FILES or sum(p.lstat().st_size for p in selected.values()) > LIMIT_BYTES:
        raise ValueError('I refuse an excessive SDK input closure before hashing')
    rows = [row(name, selected[name]) for name in sorted(selected)]
    if len(rows) > LIMIT_FILES or sum(r['size'] for r in rows) > LIMIT_BYTES:
        raise ValueError('I refuse an excessive SDK input closure')
    identity, data = manifest(rows)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError('I refuse an oversized SDK manifest')
    parent = prefix / 'lib/nanolang/sdk'
    ensure_dirs(parent)
    ensure_dirs(prefix / 'bin')
    target = parent / identity
    commands = PUBLIC + (('nano_as_capture.so',) if 'bin/nano_as_capture.so' in selected else ())
    # I check every existing public command before publishing or replacing one.
    for name in commands:
        destination = prefix / 'bin' / name
        if not os.path.lexists(destination):
            continue
        if not destination.is_symlink():
            raise ValueError('I refuse to replace an unowned compiler command')
        parts = Path(os.readlink(destination)).parts
        if (len(parts) != 7 or parts[:4] != ('..', 'lib', 'nanolang', 'sdk') or
                parts[5:] != ('bin', name) or len(parts[4]) != 64 or
                any(c not in '0123456789abcdef' for c in parts[4])):
            raise ValueError('I refuse to replace an unowned command symlink')
        old = parent / parts[4]
        if old.is_symlink():
            raise ValueError('I refuse an indirect existing SDK generation')
        verify(old, parts[4])
    stage = Path(tempfile.mkdtemp(prefix='.stage-', dir=parent))
    stage_identity = stage.stat()
    committed = False
    try:
        for r in rows:
            output = stage / r['path']
            output.parent.mkdir(parents=True, exist_ok=True)
            with selected[r['path']].open('rb') as src, output.open('xb') as dst:
                remaining = r['size']
                while remaining:
                    data_part = src.read(min(remaining, 1024 * 1024))
                    if not data_part:
                        raise ValueError('I refuse a shortened SDK copy input')
                    dst.write(data_part)
                    remaining -= len(data_part)
                if src.read(1):
                    raise ValueError('I refuse a grown SDK copy input')
                os.fchmod(dst.fileno(), r['mode'])
                dst.flush()
                os.fsync(dst.fileno())
        with (stage / MANIFEST).open('xb') as f:
            os.fchmod(f.fileno(), 0o644)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        verify(stage, identity)
        for directory, _, _ in os.walk(stage, topdown=False):
            os.chmod(directory, 0o755)
            sync_dir(directory)
        try:
            rename_exclusive(stage, target)
            committed = True
        except OSError as e:
            if e.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if target.is_symlink():
                raise ValueError('I refuse an unexpected SDK generation symlink')
            verify(target, identity)
        sync_dir(parent)
        for name in commands:
            destination = prefix / 'bin' / name
            if destination.exists() and not destination.is_symlink():
                raise ValueError('I refuse to replace an unowned compiler command')
            link = prefix / 'bin' / ('.sdk-' + name + '-' + identity)
            os.symlink('../lib/nanolang/sdk/' + identity + '/bin/' + name, link)
            try:
                os.replace(link, destination)
            finally:
                if link.is_symlink():
                    link.unlink()
        sync_dir(prefix / 'bin')
        return dict(status='installed', identity=identity, files=len(rows))
    except Exception as e:
        raise RuntimeError('I could not complete SDK installation; generation_committed=' +
                           str(committed).lower() + ': ' + str(e)) from e
    finally:
        if os.path.lexists(stage):
            current = stage.lstat()
            if (stat.S_ISDIR(current.st_mode) and
                    (current.st_dev, current.st_ino) ==
                    (stage_identity.st_dev, stage_identity.st_ino)):
                try:
                    shutil.rmtree(stage)
                except OSError as cleanup:
                    print('I retain staged SDK inputs after cleanup failure: ' +
                          str(stage) + ': ' + str(cleanup), file=sys.stderr)
            else:
                print('I retain an unknown staged SDK identity: ' + str(stage),
                      file=sys.stderr)


def uninstall(prefix):
    prefix = prefix.resolve()
    parent = owned_path(prefix, 'lib/nanolang/sdk')
    if not parent.exists():
        return dict(status='absent')
    removed = []
    for generation in sorted(parent.iterdir()):
        identity = generation.name
        if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            continue
        if generation.is_symlink() or not generation.is_dir():
            continue
        if not (generation / MANIFEST).is_file():
            continue
        actual, rows = read_manifest(generation)
        if actual != identity:
            raise ValueError('I refuse to uninstall a mismatched generation')
        # I validate the entire surviving owned set before the first deletion.
        directories = {generation}
        for r in rows:
            p = owned_path(generation, r['path'])
            if os.path.lexists(p) and row(r['path'], p) != r:
                raise ValueError('I refuse to remove a modified SDK-owned input')
            directories.update(generation / v for v in Path(r['path']).parents
                               if str(v) != '.')
        for name in PUBLIC + ('nano_as_capture.so',):
            command = owned_path(prefix, 'bin') / name
            expected = '../lib/nanolang/sdk/' + identity + '/bin/' + name
            if command.is_symlink() and os.readlink(command) == expected:
                command.unlink()
        # Unknown sentinels remain; I remove only recorded paths and their parents.
        for r in rows:
            p = owned_path(generation, r['path'])
            if os.path.lexists(p):
                p.unlink()
        (generation / MANIFEST).unlink()
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            try:
                directory.rmdir()
            except OSError as e:
                if e.errno not in (errno.ENOTEMPTY, errno.ENOENT):
                    raise
        removed.append(identity)
    return dict(status='uninstalled', generations=removed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'uninstall', 'verify'))
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--prefix', type=Path)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    if args.action == 'verify':
        identity, rows = verify(args.root, args.root.name)
        result = dict(status='verified', identity=identity, files=len(rows))
    elif args.prefix is None:
        parser.error('I require --prefix')
    elif args.action == 'install':
        result = install(args.source, args.prefix, args.source / 'scripts/native_sdk_inputs.json')
    else:
        result = uninstall(args.prefix)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
