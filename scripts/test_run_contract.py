#!/usr/bin/env python3
"""I execute explicit test arguments without changing source CLI or case selection."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def read_contract(source):
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError('I require a regular source fixture.')
    metadata = source.with_suffix('.cases.json')
    if not metadata.exists() and not metadata.is_symlink():
        return [{'name': 'default', 'args': [], 'exit_code': 0, 'contains': []}]
    if metadata.is_symlink() or not metadata.is_file() or metadata.stat().st_size > 65536:
        raise ValueError('I require a bounded regular test run contract.')
    with metadata.open('rb') as stream:
        data = stream.read(65537)
    if len(data) > 65536:
        raise ValueError('I refuse an oversized test run contract.')
    document = json.loads(data)
    if (not isinstance(document, dict) or set(document) != {'schema', 'cases'}
            or document['schema'] != 'nanolang-test-run@1'
            or not isinstance(document['cases'], list) or not 1 <= len(document['cases']) <= 16):
        raise ValueError('I require a versioned nonempty test run contract.')
    cases, names = [], set()
    for case in document['cases']:
        if not isinstance(case, dict) or set(case) - {'name', 'args', 'exit_code', 'contains'}:
            raise ValueError('I refuse unknown test run contract fields.')
        name, arguments, status = case.get('name'), case.get('args'), case.get('exit_code')
        contains = case.get('contains', [])
        if (not isinstance(name, str) or not name.isidentifier() or name in names
                or not isinstance(arguments, list) or len(arguments) > 16
                or type(status) is not int or not 0 <= status <= 255
                or not isinstance(contains, list) or len(contains) > 8
                or any(not isinstance(s, str) or not s or len(s) > 4096 or '\0' in s for s in contains)):
            raise ValueError('I require unique cases, bounded arguments and exact ordinary exit statuses.')
        names.add(name)
        resolved = []
        for argument in arguments:
            if isinstance(argument, str):
                if len(argument) > 4096 or '\0' in argument:
                    raise ValueError('I refuse an invalid literal test argument.')
                resolved.append(argument)
            elif isinstance(argument, dict) and set(argument) == {'input'}:
                value = argument['input']
                if not isinstance(value, str) or not value or '\0' in value or len(value) > 4096:
                    raise ValueError('I require a bounded test input path.')
                relative = Path(value)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('I require a relative test input within its source directory.')
                path = source.parent / relative
                canonical = path.resolve(strict=True)
                canonical.relative_to(source.parent)
                if path.is_symlink() or not canonical.is_file():
                    raise ValueError('I require a present regular test input.')
                resolved.append(str(canonical))
            else:
                raise ValueError('I require literal arguments or explicit input paths.')
        cases.append({'name': name, 'args': resolved, 'exit_code': status, 'contains': contains})
    return cases


def contains_in_region(path, start, end, needle):
    with path.open('rb') as stream:
        stream.seek(start)
        tail = b''
        while stream.tell() < end:
            data = stream.read(min(65536, end - stream.tell()))
            if not data:
                break
            block = tail + data
            if needle in block:
                return True
            tail = block[-max(1, len(needle) - 1):]
    return False


def execute(source, command, timeout, log_path):
    if not command or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('I require a command and a finite positive total run deadline.')
    cases = read_contract(source)  # I validate every case before executing any.
    deadline = time.monotonic() + timeout
    if not math.isfinite(deadline):
        raise ValueError('I cannot represent this run deadline.')
    failed = False
    with log_path.open('wb') as log:
        for case in cases:
            log.write((json.dumps({'case': case['name'], 'argv': command + case['args']}) + '\n').encode())
            log.flush()
            start = log.tell()
            remaining = deadline - time.monotonic()
            timed_out = remaining <= 0
            status = None
            if not timed_out:
                process = subprocess.Popen(command + case['args'], stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    status = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    status = process.wait()
                except BaseException:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=10)
                    raise
            end = os.fstat(log.fileno()).st_size
            matched = (not timed_out and status == case['exit_code']
                       and all(contains_in_region(log_path, start, end, text.encode()) for text in case['contains']))
            failed |= not matched
            log.write(('\n' + json.dumps({'case': case['name'], 'returncode': status,
                'expected': case['exit_code'], 'timed_out': timed_out, 'accepted': matched}) + '\n').encode())
            log.flush()
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        return execute(args.source, command, args.timeout, args.log)
    except (OSError, ValueError, TypeError, KeyboardInterrupt) as error:
        with args.log.open('ab') as log:
            log.write(('I refuse this test run contract: ' + str(error) + '\n').encode())
        return 1


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('I retain the interrupted test run log.')
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
