#!/usr/bin/env python3
"""I sample one actual Apple shadow child without replacing qualification."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from compare_cseed_source_creation import (
    ROOT, PROVIDERS, command, digest, health, shadow_trace, shared_inputs, write,
)


def stop_owned(child):
    if child.poll() is None:
        try:
            if os.getpgid(child.pid) != child.pid:
                raise RuntimeError('I refuse to stop a process outside its own diagnostic group.')
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait(timeout=10)
            return
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)


def source_inputs():
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', 'src',
        'Makefile.gnu', 'GNUmakefile', 'scripts/profile_cseed_source_shadows.py',
        '.github/workflows/cseed-source-comparison.yml'], cwd=ROOT)
    paths = [ROOT/os.fsdecode(p) for p in tracked.split(b'\0') if p]
    return {**shared_inputs(ROOT), **{str(p.relative_to(ROOT)): digest(p) for p in paths}}


def sample_creation(output, env):
    attempt = output/'attempt'
    attempt.mkdir()
    provider, source = ROOT/'bin/nanoc_c', ROOT/'src_nano/nanoc_v06.nano'
    argv = [str(provider), str(source), '--keep-c', '-o', str(attempt/'compiler')]
    record = {'argv': argv, 'cwd': str(ROOT), 'executable_sha256': digest(provider),
              'sample_attempted': False, 'native_creation_qualified': False}
    started = time.monotonic()
    observer = None
    with (attempt/'stdout').open('wb') as stdout, (attempt/'stderr').open('wb') as stderr, \
         (attempt/'sample.stdout').open('wb') as sample_out, \
         (attempt/'sample.stderr').open('wb') as sample_err:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                 start_new_session=True)
        record['parent_pid'] = child.pid
        write(attempt/'observation.json', record)
        # I consume newly appended text rather than repeatedly reading the full trace.
        with (attempt/'stderr').open('r', errors='replace') as trace:
            buffered = ''
            try:
                while child.poll() is None:
                    buffered += trace.read()
                    lines = buffered.split('\n')
                    buffered = lines.pop()
                    final_shadow = False
                    for line in lines:
                        if not line.startswith('{"event":"shadow-'):
                            continue
                        event = json.loads(line)
                        if event['event'] == 'shadow-start' and not record['sample_attempted']:
                            result = subprocess.run(['pgrep', '-P', str(child.pid)],
                                capture_output=True, text=True, timeout=5)
                            candidates = result.stdout.split()
                            if len(candidates) != 1:
                                raise RuntimeError('I require one identifiable shadow child.')
                            pid = int(candidates[0])
                            identity = subprocess.check_output(
                                ['ps', '-p', str(pid), '-o', 'ppid=,comm='], text=True, timeout=5)
                            parent, executable = identity.strip().split(None, 1)
                            if int(parent) != child.pid or Path(executable).name != 'nanoc_c':
                                raise RuntimeError('I refuse a different profiler target process.')
                            sample_argv = ['/usr/bin/sample', str(pid), '8', '10',
                                           '-file', str(attempt/'child.sample.txt')]
                            observer = subprocess.Popen(sample_argv, stdout=sample_out,
                                                        stderr=sample_err)
                            record.update(sample_attempted=True, child_pid=pid,
                                          child_identity=identity, sample_argv=sample_argv)
                            write(attempt/'observation.json', record)
                        if (event['event'] == 'shadow-complete' and event['name'] == 'main'
                                and event['source'] == str(source)):
                            final_shadow = True
                    if final_shadow:
                        record['intentional_stop'] = 'I observed the final root shadow; I stop this diagnostic before native publication.'
                        stop_owned(child)
                        break
                    if time.monotonic()-started >= 600:
                        record['diagnostic_timeout'] = True
                        stop_owned(child)
                        break
                    time.sleep(.01)
            finally:
                stop_owned(child)
                if observer is not None:
                    try:
                        record['sample_returncode'] = observer.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        record['sample_timeout'] = True
                        observer.terminate()
                        try:
                            record['sample_returncode'] = observer.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            observer.kill()
                            record['sample_returncode'] = observer.wait(timeout=5)
                record.update(returncode=child.wait(), wall_seconds=time.monotonic()-started,
                              native_product_exists=(attempt/'compiler').exists())
                record['shadow_trace'] = shadow_trace((attempt/'stderr').read_text(errors='replace'), ROOT)
                sample = attempt/'child.sample.txt'
                if sample.is_file():
                    text = sample.read_text(errors='replace')
                    record['sample'] = {'size': sample.stat().st_size, 'sha256': digest(sample),
                        'has_call_graph': 'Call graph:' in text,
                        'has_samples': bool(re.search(r'(?m)^\s+\d+\s', text.partition('Call graph:')[2]))}
                write(attempt/'observation.json', record)
    if (record.get('sample_returncode') != 0
            or not record.get('sample', {}).get('has_call_graph')
            or not record.get('sample', {}).get('has_samples')):
        raise RuntimeError('I require an actual profiler call graph; setup text is not a CPU sample.')
    if record['native_product_exists']:
        raise RuntimeError('I require my diagnostic to stop before native publication.')


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    write(output/'plan.json', {
        'source_pin': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'I diagnose one actual supervised shadow child. Sampling overhead and intentional termination do not qualify source creation, timing, platforms or release.',
        'selection': 'I retain all default dependency and root shadows.',
        'shadow_timeout_seconds': 10, 'attempts': 1, 'sample_seconds': 8,
        'sample_interval_ms': 10, 'diagnostic_prerequisite_and_creation_budget_seconds': 600,
        'native_lifecycle_admitted': False,
    })
    try:
        if sys.platform != 'darwin' or not Path('/usr/bin/sample').is_file():
            raise RuntimeError('I require my existing Apple sample tool for this diagnostic.')
        health(output, 'before-build')
        before = source_inputs()
        write(output/'inputs-before.json', before)
        env = dict(os.environ, NANO_SHADOW_TRACE='1')
        env.pop('NANO_SHADOW_TIMEOUT_SECONDS', None)
        cc = env.get('CC', 'cc')
        tool = Path(shutil.which(cc) or cc).resolve(strict=True)
        write(output/'compiler.json', {'command': cc, 'path': str(tool),
            'sha256': digest(tool),
            'version': subprocess.check_output([cc, '--version'], text=True, timeout=20),
            'environment': {k: env[k] for k in ('CC', 'CFLAGS', 'CPPFLAGS', 'LDFLAGS',
                'DEVELOPER_DIR', 'SDKROOT') if k in env}})
        build = command(['make', '-f', 'Makefile.gnu', '-j2', 'CC='+cc, 'nanoisa_dump',
                         'nano_vm', 'nvm2c', 'bin/nanoisa', 'bin/nanoc_c', 'nano_virt',
                         'nvm2c-runtime'], ROOT, output/'build.log', env, 1500)
        write(output/'build.json', build)
        if build['returncode'] or source_inputs() != before:
            raise RuntimeError('I require an actual unchanged-source provider build.')
        providers = {p: digest(ROOT/p) for p in PROVIDERS}
        write(output/'providers-before.json', providers)
        (output/'providers').mkdir()
        shutil.copy2(ROOT/'bin/nanoc_c', output/'providers/nanoc_c')
        if digest(output/'providers/nanoc_c') != providers['bin/nanoc_c']:
            raise RuntimeError('I refuse changed retained provider bytes.')
        health(output, 'before-sampling')
        try:
            sample_creation(output, env)
        finally:
            after = source_inputs()
            current = {p: digest(ROOT/p) for p in PROVIDERS}
            write(output/'inputs-after.json', after)
            write(output/'providers-after.json', current)
            if before != after or providers != current:
                raise RuntimeError('I refuse changed source or provider inputs.')
        health(output, 'after-sampling')
    except Exception as error:
        write(output/'error.json', {'type': type(error).__name__, 'message': str(error),
                                   'native_lifecycle_admitted': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output.resolve())
