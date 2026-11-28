#!/usr/bin/env python3
"""I observe one saved C-seed shadow tail; I do not qualify a release."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile

REVISION = '6e09c07a9f1651294e4669282cdc17ee14126775'
ARTIFACT = 15188575446
ZIP_SHA256 = '77326ff42549c8f0d2ed7c757a3a58611668f592348ca756e715268cb7480de9'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def captured(argv, out, label, *, timeout=300, **kwargs):
    with (out / (label + '.stdout')).open('wb') as stdout, (out / (label + '.stderr')).open('wb') as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True, **kwargs)
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if status:
            raise subprocess.CalledProcessError(status, argv)


def cache_generation_files(path):
    """I accept only complete, flat, regular-file generation contents."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f'I require a real cache generation directory: {path}')
    files = {}
    for entry in sorted(path.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f'I require regular generation entries: {entry}')
        files[entry.name] = {'size': entry.stat().st_size, 'sha256': digest(entry)}
    if not files:
        raise ValueError(f'I refuse an empty cache generation: {path}')
    return files


def restore_cache_currents(root, out):
    """I infer lost pointers only from one complete byte-equal retained tree."""
    root = root.resolve()
    out = out.resolve()
    if not out.is_relative_to(root):
        raise ValueError('I retain flattened namespace evidence inside my private root.')
    out.mkdir(parents=True, exist_ok=True)
    plans = []
    for base, local in ((root / 'modules', True), (root / 'obj/module_cache', False)):
        if not base.exists():
            continue
        if base.is_symlink() or not base.resolve().is_relative_to(root):
            raise ValueError(f'I refuse an escaped cache root: {base}')
        for current in sorted(base.rglob('current')):
            cache = current.parent
            selected = cache.name == '.build' if local else (
                cache.parent == base and re.fullmatch(r'v2-[0-9a-f]{64}', cache.name))
            if not selected:
                continue
            if not cache.resolve().is_relative_to(root):
                raise ValueError(f'I refuse an escaped cache directory: {cache}')
            relative = str(current.relative_to(root))
            if current.is_symlink():
                target = os.readlink(current)
                if not re.fullmatch(r'\.nano-gen-[A-Za-z0-9]{6}', target):
                    raise ValueError(f'I require a safe relative cache pointer: {current}')
                files = cache_generation_files(cache / target)
                plans.append({'current': relative, 'target': target, 'files': files,
                              'basis': 'existing-safe-relative-link', 'restored': False,
                              'backup': None})
                continue
            files = cache_generation_files(current)
            matches = []
            for candidate in sorted(cache.iterdir()):
                if re.fullmatch(r'\.nano-gen-[A-Za-z0-9]{6}', candidate.name):
                    if cache_generation_files(candidate) == files:
                        matches.append(candidate.name)
            if len(matches) != 1:
                raise ValueError(f'I require one complete byte-equal generation for {current}; I found {len(matches)}')
            backup = out / 'flattened-cache-currents' / current.relative_to(root)
            if os.path.lexists(backup):
                raise ValueError(f'I will not replace retained namespace evidence: {backup}')
            plans.append({'current': relative, 'target': matches[0], 'files': files,
                          'basis': 'unique-complete-byte-equal-generation', 'restored': False,
                          'backup': str(backup.relative_to(root))})
    report = {'version': 1, 'complete': False, 'qualification': False,
              'original_link_metadata_available': False, 'entries': plans}
    journal = out / 'cache-current-restoration.json'
    save(journal, report)
    try:
        for row in plans:
            if row['backup'] is None:
                continue
            current = root / row['current']
            target = current.parent / row['target']
            if cache_generation_files(current) != row['files'] or cache_generation_files(target) != row['files']:
                raise ValueError(f'I refuse changed cache contents before restoration: {current}')
            backup = root / row['backup']
            backup.parent.mkdir(parents=True, exist_ok=True)
            current.rename(backup)
            try:
                current.symlink_to(row['target'], target_is_directory=True)
            except BaseException:
                if not os.path.lexists(current):
                    backup.rename(current)
                raise
            if os.readlink(current) != row['target'] or cache_generation_files(current.resolve()) != row['files']:
                raise ValueError(f'I cannot verify my reconstructed pointer: {current}')
            row['restored'] = True
            save(journal, report)
        report['complete'] = True
    except BaseException as error:
        report['exception'] = repr(error)
        raise
    finally:
        save(journal, report)
    return report


def restore(root, out):
    archive = out / 'original-artifact.zip'
    captured(['gh', 'api', f'repos/jordanhubbard/nanolang/actions/artifacts/{ARTIFACT}/zip'],
             out, 'download')
    # I retain the response as bytes; no token or signed redirect URL is printed.
    (out / 'download.stdout').rename(archive)
    assert digest(archive) == ZIP_SHA256
    with zipfile.ZipFile(archive) as stream:
        names = set()
        prefix = Path('nanolang/nanolang')
        for entry in stream.infolist():
            name = Path(entry.filename)
            assert entry.orig_filename == entry.filename and not name.is_absolute()
            assert '..' not in name.parts and str(name) not in names
            names.add(str(name))
            assert (entry.external_attr >> 16) & 0o170000 != 0o120000
            if name.is_relative_to(prefix):
                relative = name.relative_to(prefix)
                if not relative.parts:
                    assert entry.is_dir()
                    continue
                assert relative.parts[0] in ('.test_output', 'bin', 'obj', 'modules'), relative
                target = root / relative
                assert target.resolve().is_relative_to(root)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with stream.open(entry) as source, target.open('wb') as destination:
                        shutil.copyfileobj(source, destination)
        assert 'nanolang/nanolang/bin/nanoc_bootstrap.json' in names
    for name in ('nanoc_c', 'nano_virt', 'nano_vm', 'nanoisa', 'nvm2c', 'nanoc_seed',
                 'nanoc_stage1', 'nanoc_stage2', 'nanoc'):
        path = root / 'bin' / name
        if path.is_file():
            path.chmod(0o700)
    for path in (root / '.test_output/retirement/selected-tools').glob('*'):
        if path.is_file():
            path.chmod(0o700)


def identities(root):
    receipt = json.loads((root / 'bin/nanoc_bootstrap.json').read_text())
    assert receipt['root'] == str(root) and receipt['stage_timeout_seconds'] == 1800
    assert receipt['shadow_timeout_seconds'] == 30 and receipt['raw_stage1_stage2_equal']
    guard = root / '.test_output/c-repair-original-gates'
    assert (guard / 'revision.txt').read_text().strip() == REVISION
    expected = {str(root / name): value for name, value in receipt['sources'].items()}
    expected.update({str(root / 'bin' / name): value for name, value in receipt['tools'].items()})
    expected.update({str(root / 'bin' / name): value for name, value in receipt['artifacts'].items()})
    expected.update(receipt['hosts'])
    expected.update(json.loads((guard / 'tools-before.json').read_text()))
    expected.update({str(root / 'bin' / name): value for name, value in
                     json.loads((guard / 'images-before.json').read_text()).items()})
    assert len(receipt['hosts']) == 6
    actual = {name: digest(Path(name)) for name in expected}
    assert actual == expected
    return actual


def product_tree(root, out):
    original = (root / '.test_output/c-repair-original-gates/source-tree.txt').read_text()
    current = subprocess.check_output(['git', 'ls-tree', '-r', 'HEAD'], cwd=root, text=True)
    allowed = {'.github/workflows/cseed-shadow-lifecycle.yml',
               'scripts/ci_darwin_shadow_lifecycle.c', 'scripts/ci_darwin_shadow_lifecycle.py',
               'tests/test_cseed_shadow_lifecycle_restore.py'}

    def selected(text):
        result = {}
        for line in text.splitlines():
            metadata, path = line.split('\t', 1)
            if not path.startswith('docs/') and path not in allowed:
                result[path] = metadata
        return result

    before, after = selected(original), selected(current)
    assert before == after
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=root, check=True)
    save(out / 'product-tree-guard.json', {'original_product_revision': REVISION,
         'observer_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
         'unchanged_product_entries': len(before), 'matches': True, 'qualification': False})


def controls(binary, out):
    summaries = []
    for kind in ('success', 'late', 'blocked'):
        path = out / ('control-' + kind + '.jsonl')
        captured([str(binary), '--control', kind, str(path)], out, 'control-' + kind, timeout=5)
        records = rows(path)
        report = records[-1]
        assert report['event'] == 'control' and report['kind'] == kind and report['passed']
        if kind == 'blocked':
            assert os.WIFSIGNALED(report['wait_status']) and os.WTERMSIG(report['wait_status']) == signal.SIGKILL
        else:
            assert os.WIFEXITED(report['wait_status']) and os.WEXITSTATUS(report['wait_status']) == 0
        samples = [r for r in records if r['event'] == 'sample' and r['usage_result'] == 0]
        assert samples and samples[0]['exit_absolute'] == 0
        assert all(r['start_absolute'] == samples[0]['start_absolute'] for r in samples)
        events = [r for r in records if r['event'] == 'process-event']
        marker = report['marker_ns'] - report['origin_ns']
        assert marker >= 20_000_000
        armed = [r for r in records if r['event'] == 'armed']
        triggers = [r for r in records if r['event'] == 'trigger']
        assert len(armed) == len(triggers) == 1
        assert armed[0]['monotonic_ns'] <= report['marker_ns'] <= triggers[0]['monotonic_ns']
        if kind == 'blocked':
            end = [r for r in records if r['event'] == 'end']
            assert len(end) == 1 and end[0]['monotonic_ns'] - triggers[0]['monotonic_ns'] >= 100_000_000
        if kind == 'blocked':
            assert not events and not any(r['exit_absolute'] for r in samples)
        else:
            assert len(events) == 1 and any(r['exit_absolute'] for r in samples)
            elapsed = events[0]['after_ns'] - report['marker_ns']
            assert elapsed > 0
            assert (elapsed >= 150_000_000) if kind == 'late' else (elapsed < 100_000_000)
        summaries.append({'kind': kind, 'passed': True, 'marker_after_origin_ns': marker,
                          'samples': len(samples), 'process_events': len(events),
                          'event_after_marker_ns': None if kind == 'blocked' else elapsed})
    save(out / 'controls.json', summaries)


def observe(root, out, binary):
    assert os.environ.get('NANO_SHADOW_TIMEOUT_SECONDS', '10') == '10'
    environment = dict(os.environ, NANO_SHADOW_TRACE='1', NANO_SHADOW_PHASE_TRACE='1')
    environment.pop('GH_TOKEN', None)
    environment.pop('GITHUB_TOKEN', None)
    command = [str(root / 'bin/nanoc_c'), 'tests/selfhost_shadow_emitter.nano',
               '-o', str(out / 'emitter-diagnostic')]
    result = {'qualification': False, 'whole_release': False, 'argv': command,
              'command_timeout_seconds': 300, 'shadow_seconds': 10, 'terminal': False,
              'module_cache_environment': environment.get('NANO_BUILD_CACHE')}
    sampler = None
    parent = None
    position = 0
    pending = b''
    started = time.monotonic()
    error_path = out / 'compiler.stderr'
    try:
        with (out / 'compiler.stdout').open('wb') as stdout, error_path.open('wb') as stderr:
            parent = subprocess.Popen(command, cwd=root, env=environment, stdout=stdout,
                                      stderr=stderr, start_new_session=True)
            result['compiler_pid'] = parent.pid
            save(out / 'result.json', result)
            while parent.poll() is None:
                if time.monotonic() - started > 300:
                    result['command_timeout'] = True
                    os.killpg(parent.pid, signal.SIGKILL)
                    parent.wait()
                    break
                with error_path.open('rb') as stream:
                    stream.seek(position)
                    pending += stream.read()
                    position = stream.tell()
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if not line.startswith(b'{"event":"cseed-shadow-phase"'):
                        continue
                    phase = json.loads(line)
                    if phase['phase'] == 'dispatch-start':
                        assert sampler is None
                        result['shadow_pid'] = phase['pid']
                        sampler_log = (out / 'observer.stderr').open('wb')
                        sampler = subprocess.Popen([str(binary), str(phase['pid']), str(parent.pid),
                                                    '3000', str(out / 'lifecycle.jsonl')],
                                                   stdin=subprocess.PIPE, stdout=sampler_log,
                                                   stderr=sampler_log, start_new_session=True)
                        sampler_log.close()
                        result['observer_pid'] = sampler.pid
                        save(out / 'result.json', result)
                    if phase['phase'] == 'flush-end' and sampler and sampler.poll() is None:
                        sampler.stdin.write(b'g')
                        sampler.stdin.flush()
                        sampler.stdin.close()
                        result['trigger_phase'] = phase
                        save(out / 'result.json', result)
                time.sleep(0.005)
            result['compiler_exit_code'] = parent.wait()
        if sampler:
            # EOF cancels an observer if the original supervisor stopped before flush.
            if sampler.stdin and not sampler.stdin.closed:
                sampler.stdin.close()
            result['observer_exit_code'] = sampler.wait(timeout=5)
        result['flush_observed'] = 'trigger_phase' in result
    except BaseException as error:
        result['exception'] = repr(error)
        raise
    finally:
        for process in (parent, sampler):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        result['elapsed_seconds'] = time.monotonic() - started
        result['terminal'] = True
        save(out / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    assert sys.platform == 'darwin'
    root = args.root.resolve()
    out = root / '.test_output/shadow-lifecycle'
    out.mkdir(parents=True, exist_ok=True)
    sources = ['scripts/ci_darwin_shadow_lifecycle.c', 'scripts/ci_darwin_shadow_lifecycle.py',
               '.github/workflows/cseed-shadow-lifecycle.yml',
               'tests/test_cseed_shadow_lifecycle_restore.py']
    saved = out / 'observer-sources'
    saved.mkdir()
    source_hashes = {name: digest(root / name) for name in sources}
    for name in sources:
        shutil.copyfile(root / name, saved / Path(name).name)
    save(out / 'observer-sources.json', source_hashes)
    current = root / '.test_output/retirement/capture-toolchain.json'
    shutil.copyfile(current, out / 'current-toolchain-before-restore.json')
    restore(root, out)
    restore_cache_currents(root, out)
    product_tree(root, out)
    before = identities(root)
    save(out / 'identities-before.json', before)
    libraries = {str(p): digest(p) for base in (root / 'modules', root / 'obj/module_cache')
                 for p in base.rglob('*') if p.is_file() and p.suffix in ('.so', '.dylib')}
    save(out / 'host-libraries-before.json', libraries)
    binary = out / 'lifecycle-observer'
    command = [*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-Wall', '-Wextra',
               '-Werror', '-O2', str(root / 'scripts/ci_darwin_shadow_lifecycle.c'),
               '-lproc', '-o', str(binary)]
    captured(command, out, 'build-observer', cwd=root, timeout=120)
    save(out / 'observer-build.json', {'argv': command, 'sha256': digest(binary),
                                      'source_sha256': digest(root / 'scripts/ci_darwin_shadow_lifecycle.c')})
    controls(binary, out)
    try:
        result = observe(root, out, binary)
    finally:
        after = {name: digest(Path(name)) for name in before}
        save(out / 'identities-after.json', after)
        assert before == after
        assert source_hashes == {name: digest(root / name) for name in sources}
        after_libraries = {str(p): digest(p) for base in (root / 'modules', root / 'obj/module_cache')
                           for p in base.rglob('*') if p.is_file() and p.suffix in ('.so', '.dylib')}
        save(out / 'host-libraries-after.json', after_libraries)
        assert all(after_libraries.get(name) == value for name, value in libraries.items())
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=root, check=True)
    assert result['flush_observed'] and result.get('observer_exit_code') == 0, result


if __name__ == '__main__':
    main()
