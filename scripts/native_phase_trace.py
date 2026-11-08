#!/usr/bin/env python3
"""I observe the real native publisher tools without replacing their behavior."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def file_record(path, destination=None):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    record = {'path': str(path.resolve()), 'size': path.stat().st_size, 'sha256': digest.hexdigest()}
    if destination is not None:
        shutil.copyfile(path, destination)
        record['copy'] = str(destination)
        if file_record(destination)['sha256'] != record['sha256']:
            raise RuntimeError('I could not retain exact native phase bytes.')
    return record


def write_record(path, record):
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(record, indent=2)+'\n')
    partial.replace(path)


def configure(work, environment, root=ROOT):
    """I wrap only the self-hosted publisher's existing selected executables."""
    environment = dict(environment)
    directory = Path(work)/'native-phase-trace'
    directory.mkdir()
    selected = {
        'cc': shlex.split(environment.get('NANO_CC') or environment.get('CC') or 'cc'),
        'nvm2c': [environment.get('NANO_NVM2C') or str(root/'bin/nvm2c')],
    }
    for role, argv in selected.items():
        if not argv:
            raise ValueError('I require a selected native phase tool.')
        executable = shutil.which(argv[0], path=environment.get('PATH'))
        if executable is None:
            raise ValueError(f'I cannot resolve my selected {role} tool.')
        argv[0] = executable
        config = directory/(role+'.json')
        config.write_text(json.dumps({'schema': 'nanolang.native-phase-trace.v1',
            'role': role, 'tool_argv': argv, 'directory': str(directory.resolve())}, indent=2)+'\n')
        wrapper = directory/(role+'-wrapper')
        command = [sys.executable, str(root/'scripts/native_phase_trace.py'), '--config', str(config.resolve()), '--']
        wrapper.write_text('#!/bin/sh\nexec '+' '.join(shlex.quote(value) for value in command)+' "$@"\n')
        wrapper.chmod(0o700)
        environment['NANO_CC' if role == 'cc' else 'NANO_NVM2C'] = (
            shlex.quote(str(wrapper.resolve())) if role == 'cc' else str(wrapper.resolve()))
    recorder = directory/'marker-recorder'
    marker_command = [sys.executable, str(root/'scripts/native_phase_trace.py'),
                      '--directory', str(directory.resolve()), '--mark']
    recorder.write_text('#!/bin/sh\nexec '+' '.join(shlex.quote(value) for value in marker_command)+' "$@"\n')
    recorder.chmod(0o700)
    environment['NANO_COMPILER_PHASE_RECORDER'] = str(recorder.resolve())
    return environment


def summarize(work):
    directory = Path(work)/'native-phase-trace'
    phases = [json.loads(path.read_text()) for path in sorted(directory.glob('phase-*/phase.json'))]
    summary = {'schema': 'nanolang.native-phase-summary.v1',
        'native_lifecycle_admitted': False, 'phase_count': len(phases), 'phases': phases,
        'driver_markers': sorted([json.loads(path.read_text()) for path in directory.glob('marker-*.json')],
                                 key=lambda row: row['monotonic_seconds']),
        'scope': 'I retain publisher tool calls and opt-in driver boundaries; other elapsed time remains unattributed.'}
    write_record(Path(work)/'native-phase-summary.json', summary)
    return summary


def run_phase(config, arguments):
    if config.get('schema') != 'nanolang.native-phase-trace.v1' or config.get('role') not in ('cc', 'nvm2c'):
        raise ValueError('I require my native phase trace configuration.')
    argv = config['tool_argv']+arguments
    directory = Path(tempfile.mkdtemp(prefix='phase-', dir=config['directory']))
    status_file = directory/'phase.json'
    record = {'role': config['role'], 'argv': argv, 'cwd': os.getcwd(),
        'observer_pid': os.getpid(), 'parent_pid': os.getppid(), 'completed': False,
        'tool': file_record(argv[0]), 'observer': file_record(__file__),
        'inputs_before': [], 'outputs': [], 'observer_started_monotonic': time.monotonic()}
    write_record(status_file, record)
    output = None
    inputs = []
    for index, argument in enumerate(arguments):
        if index and arguments[index-1] == '-o':
            output = argument
            continue
        direct_input = (config['role'] == 'cc' and not argument.startswith('-'))
        if ((config['role'] == 'nvm2c' and index == 0) or direct_input):
            if Path(argument).is_file():
                inputs.append(argument)
                copied = file_record(argument, directory/('input-'+str(len(inputs))+Path(argument).suffix))
                record['inputs_before'].append(copied)
    write_record(status_file, record)
    started = time.monotonic()
    process = subprocess.Popen(argv)
    record.update(tool_pid=process.pid, tool_started_monotonic=started)
    if hasattr(os, 'getpgid'):
        try:
            record['tool_process_group'] = os.getpgid(process.pid)
        except ProcessLookupError:
            record['tool_process_group'] = None
    write_record(status_file, record)
    code = process.wait()
    record.update(completed=True, returncode=code, tool_elapsed_seconds=time.monotonic()-started,
                  inputs_after=[file_record(path) for path in inputs], tool_after=file_record(argv[0]))
    record['inputs_unchanged'] = all(after is not None and before['sha256'] == after['sha256']
        for before, after in zip(record['inputs_before'], record['inputs_after']))
    if output is not None and Path(output).is_file():
        record['outputs'].append(file_record(output, directory/('output'+Path(output).suffix)))
    record['observer_elapsed_seconds'] = time.monotonic()-record['observer_started_monotonic']
    write_record(status_file, record)
    return code if code >= 0 else 128-code


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--config', type=Path)
    selection.add_argument('--mark')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.mark is not None:
        if options.directory is None:
            parser.error('I require a marker directory.')
        record = {'schema': 'nanolang.compiler-phase-marker.v1', 'phase': options.mark,
                  'monotonic_seconds': time.monotonic(), 'observer_pid': os.getpid(),
                  'parent_pid': os.getppid()}
        descriptor, name = tempfile.mkstemp(prefix='marker-', suffix='.partial', dir=options.directory)
        os.close(descriptor)
        write_record(Path(name).with_suffix('.json'), record)
        return 0
    arguments = options.arguments[1:] if options.arguments[:1] == ['--'] else options.arguments
    return run_phase(json.loads(options.config.read_text()), arguments)


if __name__ == '__main__':
    raise SystemExit(main())
