#!/usr/bin/env python3
"""I qualify a fresh Stage 1 bootstrap at an absolute worktree path of at least 300 characters."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or not 60 <= args.timeout <= 7200:
        parser.error('I require 1..8 build jobs and a 60..7200-second deadline.')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=ROOT, check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    parent = Path.home()/'.worktrees/nanolang'
    parent.mkdir(parents=True, exist_ok=True)
    retained = Path(tempfile.mkdtemp(prefix='qualify-long-path-', dir=parent)).resolve()
    worktree = retained/'source'
    while len(str(worktree)) < 300:
        worktree /= 'native-bootstrap-path-' + 'x'*40
    record = {'schema': 'nanolang/long-path-bootstrap-check@1', 'source_commit': revision,
              'worktree': str(worktree), 'path_characters': len(str(worktree)),
              'path_bytes': len(os.fsencode(worktree)), 'steps': [], 'passed': False,
              'retained_execution_admission': False, 'shadow_timeout_seconds': 120}
    report = retained/'result.json'

    def run(argv, cwd):
        step = {'argv': [str(a) for a in argv], 'cwd': str(cwd),
                'started': datetime.now(timezone.utc).isoformat()}
        log = retained/f'{len(record["steps"]):02d}.log'
        step['log'] = str(log)
        record['steps'].append(step)
        report.write_text(json.dumps(record, indent=2)+'\n')
        print('I run ' + ' '.join(step['argv']), flush=True)
        environment = os.environ.copy()
        for name in ('MAKEFLAGS', 'MFLAGS', 'MAKEOVERRIDES'):
            environment.pop(name, None)
        with log.open('w') as output:
            process = subprocess.Popen(step['argv'], cwd=cwd, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True, env=environment)
            try:
                status = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                status = 124
            step['returncode'] = status
            step['finished'] = datetime.now(timezone.utc).isoformat()
            report.write_text(json.dumps(record, indent=2)+'\n')
        if status:
            raise RuntimeError(f'I retain the failed command and output in {log}.')

    print(f'I retain my long-path qualification in {retained}.', flush=True)
    try:
        run(['git', 'worktree', 'add', '--detach', '--no-checkout', worktree, revision], ROOT)
        # I omit only historical evidence. All compiler/module/build/test inputs
        # remain at their original paths; no old objects or receipts are copied.
        run(['git', 'sparse-checkout', 'set', '--no-cone', '/*', '!/docs/evidence/'], worktree)
        run(['git', 'checkout', '--detach', revision], worktree)
        if len(str(worktree.resolve())) < 300:
            raise RuntimeError('I require an actual absolute worktree path of at least 300 characters.')
        run(['make', f'-j{args.jobs}', 'OBJ_DIR=obj', 'BIN_DIR=bin',
             f'NANO_BUILD_CACHE={worktree}/obj/module_cache',
             'NANO_SHADOW_TIMEOUT_SECONDS=120', 'bootstrap1'], worktree)
        manifest = json.loads((worktree/'bin/nanoc_bootstrap.json').read_text())
        if manifest.get('phase') != 'stage1' or manifest.get('root') != str(worktree.resolve()):
            raise RuntimeError('I require a fresh Stage 1 manifest rooted at this long worktree path.')
        if not (worktree/'.bootstrap1.built').is_file():
            raise RuntimeError('I require the successful Stage 1 Make checkpoint.')
        record['bootstrap_manifest'] = str(worktree/'bin/nanoc_bootstrap.json')
        record['passed'] = True
        print('I passed a fresh Stage 1 bootstrap at the retained long path.', flush=True)
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        record['error'] = str(error)
        print(str(error), flush=True)
    finally:
        report.write_text(json.dumps(record, indent=2)+'\n')
        print(f'I retain my result in {report}.', flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
