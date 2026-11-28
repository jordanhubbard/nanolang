#!/usr/bin/env python3
"""I observe one retained Stage2 command at its original deadline."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

REVISION = '54a43a9af9917428a6b9840c7a44cf789eaf227d'
ARTIFACT = 15246228880
ZIP_SHA256 = 'f18398bb874ea9bb3effa1dde0b36e44dab824211255807657d9696fc9b57aac'
RESTORE_SHA256 = '5693f86ea998efff0abde49284d362c79c21556cabc6739647438878ce81771f'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def observe(command, directory, *, cwd, env=None, deadline=300, interval=1, cpu_sampler=None):
    """I retain both original pipes and sample only my new process group."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    errors = []
    process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    start = time.monotonic()
    result = {'qualification': False, 'argv': list(map(str, command)),
              'pid': process.pid, 'deadline_seconds': deadline,
              'command_timeout': False, 'output_complete': True, 'terminal': False}
    save(directory / 'result.json', result)

    def sample():
        try:
            with (directory / 'processes.jsonl').open('w') as stream:
                while not stop.is_set():
                    rows = subprocess.run(
                        ['ps', '-axo', 'pid=,ppid=,pgid=,stat=,time=,comm='],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, timeout=2, check=True)
                    owned = []
                    for line in rows.stdout.splitlines():
                        fields = line.split(None, 5)
                        if len(fields) == 6 and int(fields[2]) == process.pid:
                            owned.append({'pid': int(fields[0]), 'parent': int(fields[1]),
                                          'group': int(fields[2]), 'state': fields[3],
                                          'cpu': fields[4], 'command': fields[5]})
                    stream.write(json.dumps({'elapsed_seconds': time.monotonic() - start,
                                             'processes': owned}) + '\n')
                    stream.flush()
                    stop.wait(interval)
        except BaseException as error:
            errors.append(repr(error))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    cpu_session = None
    try:
        if cpu_sampler is not None:
            cpu_session = cpu_sampler.start(process.pid, directory / 'cpu-profile')
        try:
            stdout, stderr = process.communicate(
                timeout=max(0, deadline - (time.monotonic() - start)))
        except subprocess.TimeoutExpired:
            result['command_timeout'] = True
            result['deadline_observed_seconds'] = time.monotonic() - start
            os.killpg(process.pid, signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired as error:
                stdout, stderr = error.output or b'', error.stderr or b''
                result['output_complete'] = False
                process.wait(timeout=5)
                process.stdout.close()
                process.stderr.close()
        result.update(exit_code=process.returncode,
                      elapsed_seconds=time.monotonic() - start)
        (directory / 'stdout').write_bytes(stdout)
        (directory / 'stderr').write_bytes(stderr)
        result.update(stdout_sha256=digest(directory / 'stdout'),
                      stderr_sha256=digest(directory / 'stderr'))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        stop.set()
        thread.join(timeout=3)
        result.update(terminal=True, observer_errors=errors,
                      sampler_stopped=not thread.is_alive())
        save(directory / 'result.json', result)
        if cpu_session is not None:
            result['cpu_profile'] = cpu_sampler.finish(cpu_session)
            save(directory / 'result.json', result)
    return result


def controls(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cases = [('complete', 'print("complete", flush=True)', 0, False),
             ('nonzero', 'import sys; print("refused", file=sys.stderr); sys.exit(23)', 23, False),
             ('timeout', 'import signal; print("armed", flush=True); signal.pause()', -9, True)]
    results = []
    for name, source, expected, timeout in cases:
        result = observe([sys.executable, '-c', source], directory / name,
                         cwd=directory, deadline=0.2 if timeout else 5, interval=0.02)
        assert result['exit_code'] == expected and result['command_timeout'] == timeout
        assert result['terminal'] and result['output_complete']
        assert result['sampler_stopped'] and not result['observer_errors']
        if timeout:
            assert (directory / name / 'stdout').read_bytes() == b'armed\n'
        results.append(result)
    save(directory / 'controls.json', results)
    return results


def original_command(root, log, temporary):
    matches = re.findall(r"^subprocess.TimeoutExpired: Command '(\[.*\])' timed out after 300 seconds$",
                         log, re.MULTILINE)
    assert len(matches) == 1
    command = ast.literal_eval(matches[0])
    assert isinstance(command, list) and len(command) == 4
    assert command[:3] == [str(root / 'bin/nanoc_stage2'),
                           str(root / 'tests/selfhost_shadow_emitter.nano'), '-o']
    output = Path(command[3])
    assert output.is_absolute() and output.name == 'nanoc_stage2'
    assert '..' not in output.parts and output.resolve() == output
    assert output.parent != temporary and output.parent.is_relative_to(temporary)
    assert output.parent.name.startswith('nano-native-shadows-')
    assert not output.parent.exists()
    return command


def retain_temporaries(directory, destination):
    """I copy regular compiler intermediates and describe links without following them."""
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    assert not destination.is_relative_to(directory)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in sorted(directories + files):
            source = Path(parent) / name
            relative = source.relative_to(directory)
            if source.is_symlink():
                records.append({'path': str(relative), 'link': os.readlink(source)})
            elif source.is_file():
                target = destination / 'tree' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                before = digest(source)
                assert digest(target) == before
                records.append({'path': str(relative), 'size': source.stat().st_size,
                                'sha256': before})
            elif not source.is_dir():
                records.append({'path': str(relative), 'nonregular_mode': source.lstat().st_mode})
    save(destination / 'files.json', records)
    return records


def restore_admitted_inputs(helper, root, out):
    helper.restore(root, out)
    report = helper.restore_cache_currents(root, out)
    assert report['complete'] and report['entries']
    assert (out / 'cache-current-restoration.json').is_file()
    for entry in report['entries']:
        current = root / entry['current']
        assert current.is_symlink() and os.readlink(current) == entry['target']
        assert current.resolve().is_dir()
    helper.product_tree(root, out)
    return helper.identities(root)


def native_control(root, out):
    directory = out / 'native-control'
    directory.mkdir(parents=True, exist_ok=False)
    source, binary = directory / 'source.nano', directory / 'program'
    source.write_text('fn diagnostic_answer() -> int { return 7 }\n'
                      'shadow diagnostic_answer { assert (== (diagnostic_answer) 7) }\n'
                      'fn main() -> int { return (- (diagnostic_answer) 7) }\n'
                      'shadow main { assert (== (main) 0) }\n')
    temporary = directory / 'tmp'
    temporary.mkdir()
    env = dict(os.environ, TMPDIR=str(temporary), NANO_TRACE_BUILD='1')
    compiled = observe([root / 'bin/nanoc_stage2', source, '-o', binary],
                       directory / 'compile', cwd=root, env=env, deadline=60)
    assert compiled['exit_code'] == 0 and not compiled['command_timeout']
    assert compiled['output_complete'] and not compiled['observer_errors']
    assert compiled['sampler_stopped'] and binary.read_bytes()[:4] == b'\x7fELF'
    executed = observe([binary], directory / 'execute', cwd=root, env=env, deadline=5)
    assert executed['exit_code'] == 0 and not executed['command_timeout']
    assert executed['output_complete'] and not executed['observer_errors']
    assert executed['sampler_stopped']
    save(directory / 'control.json', {'qualification': False, 'compile': compiled,
                                     'execute': executed, 'binary_sha256': digest(binary),
                                     'control_build_trace': True})
    return compiled, executed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--restore-helper', type=Path, required=True)
    parser.add_argument('--sample-cpu', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    assert sys.platform == 'linux'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == REVISION
    assert digest(args.restore_helper) == RESTORE_SHA256
    out = root / '.test_output/stage2-emitter-diagnostic'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'observer.py').write_bytes(Path(__file__).read_bytes())
    (out / 'restore-helper.py').write_bytes(args.restore_helper.read_bytes())
    observer_sources = [Path(__file__), args.restore_helper,
                        Path(__file__).parents[1] / 'tests/test_stage2_emitter_diagnostic.py',
                        Path(__file__).parents[1] / 'tests/test_cseed_shadow_lifecycle_restore.py']
    source_hashes = {str(path): digest(path) for path in observer_sources}
    save(out / 'observer-sources-before.json', source_hashes)
    saved_sources = out / 'controller-sources'
    saved_sources.mkdir()
    for path in observer_sources:
        (saved_sources / path.name).write_bytes(path.read_bytes())
    spec = importlib.util.spec_from_file_location('retained_restore', args.restore_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.REVISION, helper.ARTIFACT, helper.ZIP_SHA256 = REVISION, ARTIFACT, ZIP_SHA256
    controls(out / 'controls')
    cpu_sampler = None
    if args.sample_cpu:
        from stage2_cpu_sampler import Sampler, calibrate
        observer_sources.append(Path(__file__).with_name('stage2_cpu_sampler.py'))
        path = observer_sources[-1]
        source_hashes[str(path)] = digest(path)
        (saved_sources / path.name).write_bytes(path.read_bytes())
        save(out / 'observer-sources-before.json', source_hashes)
        calibrate(out / 'cpu-sampler-control')
        cpu_sampler = Sampler()
    before = restore_admitted_inputs(helper, root, out)
    save(out / 'identities-before.json', before)
    native_control(root, out)
    after_control = helper.identities(root)
    save(out / 'identities-after-control.json', after_control)
    assert after_control == before
    log = (root / '.test_output/c-repair-original-gates/native-suites.log').read_text()
    temporary = Path(os.environ['RUNNER_TEMP']).resolve()
    command = original_command(root, log, temporary)
    directory = Path(command[-1]).parent
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TMPDIR=str(directory))
    receipt = json.loads((root / 'bin/nanoc_bootstrap.json').read_text())
    assert shlex.split(env.get('CC', 'cc')) == receipt['compiler']
    assert env.get('NANO_SHADOW_TIMEOUT_SECONDS', '10') == '10'
    assert not env.get('NANO_SHADOW_TRACE') and not env.get('NANO_SHADOW_PHASE_TRACE')
    save(out / 'invocation.json', {'qualification': False, 'product_revision': REVISION,
                                  'artifact': ARTIFACT, 'archive_sha256': ZIP_SHA256,
                                  'argv': command, 'TMPDIR': str(directory),
                                  'observer_revision': os.environ.get('GITHUB_SHA'),
                                  'selected_environment': {name: env.get(name) for name in
                                      ['CC', 'CFLAGS', 'LDFLAGS', 'COMPILER_PATH',
                                       'NANO_SHADOW_TIMEOUT_SECONDS', 'NANO_SHADOW_TRACE',
                                       'NANO_SHADOW_PHASE_TRACE']},
                                  'original_argv_preserved': True,
                                  'terminal_artifact_state_not_original_precommand_state': True})
    try:
        result = observe(command, out / 'command', cwd=root, env=env, deadline=300,
                         cpu_sampler=cpu_sampler)
    finally:
        after = helper.identities(root)
        save(out / 'identities-after.json', after)
        assert before == after
        helper.product_tree(root, out)
        actual_sources = {name: digest(name) for name in source_hashes}
        save(out / 'observer-sources-after.json', actual_sources)
        assert source_hashes == actual_sources
        retain_temporaries(directory, out / 'product-temporaries')
    assert result['output_complete'] and result['sampler_stopped'] and not result['observer_errors']
    save(out / 'diagnostic.json', {'qualification': False, 'whole_release': False,
                                  'observation_valid': True, 'guards_match': True,
                                  'namespace_recovery_complete': True,
                                  'native_compile_and_run_control_passed': True,
                                  'controls_passed': 3, 'product': result})


if __name__ == '__main__':
    main()
