#!/usr/bin/env python3
"""I require working ASan use-after-return and UBSan detection before CI builds."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from ci_sanitizer_partitions import CC, CFLAGS, LDFLAGS

ROOT = Path(__file__).resolve().parents[1]


def check(compiler, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = output / 'canary'
    compile_command = [*shlex.split(compiler), *shlex.split(CFLAGS),
                       '-fno-sanitize-recover=all',
                       str(ROOT / 'tests/fixtures/sanitizer_toolchain_canary.c'),
                       '-o', str(binary), *shlex.split(LDFLAGS)]
    report = {'compiler': compiler, 'compile': compile_command, 'runs': [], 'success': False}
    # I retain detection even if an inherited environment tried to disable it.
    env = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0:detect_stack_use_after_return=1',
           'UBSAN_OPTIONS': 'halt_on_error=1'}

    def run(command, name):
        try:
            result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
        except subprocess.TimeoutExpired as error:
            for suffix, data in [('stdout', error.stdout), ('stderr', error.stderr)]:
                (output / (name + '.' + suffix)).write_bytes(data or b'')
            raise
        (output / (name + '.stdout')).write_text(result.stdout)
        (output / (name + '.stderr')).write_text(result.stderr)
        return result

    try:
        compiled = run(compile_command, 'compile')
        report['compile_returncode'] = compiled.returncode
        if compiled.returncode:
            raise RuntimeError('I could not compile my sanitizer canary.')
        for mode, diagnosis in [('valid', None),
                                ('uar', 'ERROR: AddressSanitizer: stack-use-after-return'),
                                ('overflow', 'runtime error: signed integer overflow')]:
            result = run([str(binary), mode], mode)
            accepted = (result.returncode == 0 and
                        result.stdout.strip() == 'I completed 100000 valid calls.' and
                        not result.stderr) if diagnosis is None else (
                            result.returncode != 0 and diagnosis in result.stderr)
            report['runs'].append({'mode': mode, 'returncode': result.returncode,
                                   'expected_diagnosis': diagnosis, 'accepted': accepted})
        if not all(item['accepted'] for item in report['runs']):
            raise RuntimeError('I refuse a sanitizer toolchain that misses my canary.')
        report['success'] = True
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        report['error'] = str(error)
        raise
    finally:
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('I detected use-after-return and signed overflow after 100000 valid calls.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=CC)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        check(args.cc, args.output)
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
