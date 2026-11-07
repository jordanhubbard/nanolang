#!/usr/bin/env python3
"""I retain verified successful temporary artifacts and leave failed roots intact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        info = path.lstat()
        record = {'mode': stat.S_IMODE(info.st_mode)}
        if stat.S_ISLNK(info.st_mode):
            record.update(kind='link', target=os.readlink(path))
        elif stat.S_ISDIR(info.st_mode):
            record.update(kind='directory')
        elif stat.S_ISREG(info.st_mode):
            with path.open('rb') as stream:
                record.update(kind='file', size=info.st_size, sha256=digest(stream))
        else:
            raise ValueError(f'I cannot archive this temporary file type: {path}')
        result[path.relative_to(root).as_posix()] = record
    return result


def verify_archive(archive, expected):
    observed = {}
    with tarfile.open(archive, 'r:gz') as stream:
        for entry in stream:
            if entry.name in observed:
                raise ValueError('I refuse duplicate archive entries.')
            record = {'mode': entry.mode}
            if entry.isdir():
                record.update(kind='directory')
            elif entry.issym():
                record.update(kind='link', target=entry.linkname)
            elif entry.isfile() or entry.islnk():
                with stream.extractfile(entry) as content:
                    record.update(kind='file', size=expected[entry.name]['size'], sha256=digest(content))
            else:
                raise ValueError('I refuse unsupported archive entries.')
            observed[entry.name] = record
    if observed != expected:
        raise ValueError('I refuse to remove temporary evidence whose archive differs.')


def archive_success(root, destination):
    """I only remove my freshly allocated root after two source inventories agree."""
    if root.resolve() in destination.resolve().parents:
        raise ValueError('I require an archive outside the temporary root.')
    expected = inventory(root)
    partial = destination.with_suffix(destination.suffix + '.partial')
    with tarfile.open(partial, 'w:gz', compresslevel=1, dereference=False) as stream:
        for name in expected:
            stream.add(root / name, arcname=name, recursive=False)
    verify_archive(partial, expected)
    if inventory(root) != expected:
        raise ValueError('I retain temporary evidence changed during archival.')
    with partial.open('rb') as stream:
        archive_hash = digest(stream)
        os.fsync(stream.fileno())
    partial.replace(destination)
    manifest = destination.with_suffix(destination.suffix + '.json')
    manifest.write_text(json.dumps({'archive_sha256': archive_hash, 'entries': expected}, indent=2) + '\n')
    shutil.rmtree(root)
    return archive_hash


def measure(paths, sizes=False):
    measurements = {}
    deadline = time.monotonic() + 30
    for name, raw in paths.items():
        path = Path(raw).expanduser().resolve()
        existing = path
        while not existing.exists():
            existing = existing.parent
        usage = shutil.disk_usage(existing)
        row = {'path': str(path), 'exists': path.exists(), 'device': existing.stat().st_dev,
               'total': usage.total, 'free': usage.free}
        if sizes and name != 'developer_tools' and path.exists():
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired('du sampling budget', 30)
                size = subprocess.run(['du', '-sk', str(path)], capture_output=True, text=True, timeout=min(10, remaining))
                row['size_kib'] = int(size.stdout.split()[0]) if size.returncode == 0 else None
                row['size_probe_returncode'] = size.returncode
            except (subprocess.TimeoutExpired, ValueError, OSError) as error:
                row['size_probe_error'] = type(error).__name__
        measurements[name] = row
    return measurements


def run(command, label, output, temp_parent=None, interval=30, min_free_bytes=2 * 1024**3):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='nano-qualification-', dir=temp_parent))
    # The root is fresh. I never discover old directories and infer ownership.
    paths = {'temporary': root, 'workspace': Path.cwd(), 'output': output,
             'package_cache': Path.home() / ('Library/Caches/Homebrew' if sys.platform == 'darwin' else '.cache'),
             'module_cache': Path.home() / 'Library/Developer/Xcode/DerivedData',
             'developer_tools': Path(os.environ.get('DEVELOPER_DIR', '/Applications' if sys.platform == 'darwin' else '/usr'))}
    journal = output / (root.name + '.jsonl')
    def record(event, sizes=False, **extra):
        volumes = measure(paths, sizes)
        with journal.open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'label': label, 'event': event,
                                     'temporary_root': str(root), 'volumes': volumes, **extra}) + '\n')
            stream.flush()
        return volumes
    def guard(volumes):
        if any(volumes[name]['free'] < min_free_bytes for name in ('temporary', 'workspace', 'output')):
            raise RuntimeError('I retain a two-GiB log reserve and stop new qualification writes.')
    process = None
    try:
        guard(record('before', sizes=True, command=command))
        env = dict(os.environ, TMPDIR=str(root), TMP=str(root), TEMP=str(root))
        # Nested module invocations must not put their archives inside the root I remove.
        env['NANOLANG_QUALIFICATION_ARCHIVES'] = str(output)
        print(f'I retain {label} temporary evidence at {root}', flush=True)
        process = subprocess.Popen(command, env=env, start_new_session=True)
        while True:
            try:
                status = process.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                guard(record('running'))
        # Native tests supervise their children. An unexpected surviving group
        # blocks archival; its outputs cannot yet be considered inactive.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('I retain a temporary root with surviving child processes.')
        volumes = record('after', sizes=True, returncode=status)
        if status == 0:
            guard(volumes)
            archive = output / (root.name + '.tar.gz')
            archive_hash = archive_success(root, archive)
            record('archived', archive=str(archive), archive_sha256=archive_hash)
        else:
            record('retained_failure', returncode=status)
        return status
    except BaseException as error:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        try:
            record('retained_error', error=type(error).__name__)
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'unittest'])
    parser.add_argument('--label', default='unittest')
    parser.add_argument('--output', default=os.environ.get('NANOLANG_QUALIFICATION_ARCHIVES', '.test_output/retirement/storage'))
    parser.add_argument('--temp-parent', default=os.environ.get('RUNNER_TEMP'))
    args, command = parser.parse_known_args(argv)
    if command[:1] == ['--']:
        command = command[1:]
    if args.mode == 'unittest':
        # My Make recipes pass explicit module names and -v. I refuse unknown
        # selectors rather than silently change collection or ordering.
        if command[:1] != ['-v'] or len(command) < 2 or any(
                not name.startswith('tests.') or not all(part.isidentifier() for part in name.split('.'))
                for name in command[1:]):
            parser.error('I require -v followed by explicit tests.* module names.')
        first_failure = 0
        for module in command[1:]:
            status = run([sys.executable, '-m', 'unittest', '-v', module], module,
                         args.output, args.temp_parent)
            if status and not first_failure:
                first_failure = status
        return first_failure
    if not command:
        parser.error('I require a command after --.')
    return run(command, args.label, args.output, args.temp_parent)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('I retain interrupted qualification outputs.')
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
