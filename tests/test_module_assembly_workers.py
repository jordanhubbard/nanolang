"""I bound real retained compiler overlap and reap every owned worker."""
from pathlib import Path
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKER = '''from pathlib import Path
import json, os, signal, sys
source, output = Path(sys.argv[-3]), Path(sys.argv[-1])
assert sys.argv[-2] == '-o'
key = source.stem.removeprefix('__snapshot_')
def event(kind):
    os.write(int(os.environ['TEST_WORKER_EVENTS']),
             (json.dumps(dict(kind=kind, key=key, pid=os.getpid(),
                              hidden='-fvisibility=hidden' in sys.argv))+'\\n').encode())
event('start')
assert os.read(int(os.environ['TEST_WORKER_RELEASE']), 1) == b'x'
output.write_bytes(source.read_bytes())
os.write(1, ('compiler stdout '+key+'\\n').encode())
os.write(2, ('compiler stderr '+key+'\\n').encode())
event('finish')
if key == os.environ.get('TEST_WORKER_SIGNAL'): os.kill(os.getpid(), signal.SIGTERM)
sys.exit(1 if key == os.environ.get('TEST_WORKER_FAIL') else 0)
'''


class AssemblyWorkers(unittest.TestCase):
    def run_case(self, *, limit=2, ordinary=5, shared=3, spawn=-1,
                 interrupt=False, failure=False, signaled=False, non_c=False, stem='__snapshot'):
        probe = ROOT / 'obj/test_module_assembly_workers'
        self.assertTrue(probe.is_file(), 'I require make test-module-assembly-workers.')
        total = ordinary + shared
        valid_limit = 1 <= limit <= 8
        expected = (0 if non_c or not valid_limit else spawn if spawn >= 0
                    else min(total, limit) if failure or signaled else total)
        accepted = valid_limit and spawn < 0 and not failure and not signaled
        with tempfile.TemporaryDirectory(prefix='nano-assembly-workers-') as directory:
            work = Path(directory) / 'spaces and \'quotes"'
            work.mkdir()
            worker = work / 'compiler.py'
            worker.write_text(WORKER)
            inputs = {}
            for group, count in enumerate((ordinary, shared)):
                for index in range(count):
                    path = work / f'__snapshot_{group}_{index}.i'
                    inputs[path] = f'I retain unit {group}/{index}.\n'.encode()
                    path.write_bytes(inputs[path])
            events_read, events_write = os.pipe()
            release_read, release_write = os.pipe()
            env = dict(os.environ, TEST_WORKER_EVENTS=str(events_write),
                       TEST_WORKER_RELEASE=str(release_read))
            if failure: env['TEST_WORKER_FAIL'] = '0_0'
            if signaled: env['TEST_WORKER_SIGNAL'] = '0_0'
            command = [str(probe), str(work), shlex.join([sys.executable, str(worker)]), str(limit),
                       str(ordinary), str(shared), str(spawn), str(int(interrupt)), str(int(non_c)), stem]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, pass_fds=(events_write, release_read),
                                       start_new_session=True)
            os.close(events_write)
            os.close(release_read)
            events, pending, started, released = [], b'', 0, 0
            deadline = time.monotonic() + 60
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    self.assertGreater(remaining, 0, command)
                    ready, _, _ = select.select([events_read], [], [], remaining)
                    self.assertTrue(ready, command)
                    data = os.read(events_read, 4096)
                    if not data: break
                    pending += data
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        event = json.loads(line)
                        events.append(event)
                        if event['kind'] == 'start':
                            started += 1
                            batch = min(limit, expected - released)
                            if batch > 0 and started - released == batch:
                                os.write(release_write, b'x' * batch)
                                released += batch
                stdout, stderr = process.communicate(timeout=max(.01, deadline - time.monotonic()))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=10)
                os.close(events_read)
                os.close(release_write)
            self.assertEqual(pending, b'')
            self.assertEqual(process.returncode, 0 if accepted else 1, stderr)
            lines = stdout.splitlines()
            result = json.loads(lines[-1])
            self.assertEqual(result['ok'], accepted)
            self.assertTrue(1 <= result['default_limit'] <= 8)
            self.assertEqual(result['maximum'], min(limit, expected) if limit > 1 else 0)
            self.assertEqual(result['interrupted'], interrupt and expected > 0 and limit > 1)
            starts = [event for event in events if event['kind'] == 'start']
            finishes = [event for event in events if event['kind'] == 'finish']
            self.assertEqual(len(starts), expected)
            self.assertEqual({event['key'] for event in starts}, {event['key'] for event in finishes})
            self.assertEqual(len(finishes), expected)
            for event in starts:
                group, index = map(int, event['key'].split('_'))
                self.assertEqual(event['hidden'], group == 1)
                path = work / f'{stem}_{group}_{index}.s'
                self.assertEqual(path.read_bytes(), inputs[work / f'__snapshot_{group}_{index}.i'])
                with self.assertRaises(ProcessLookupError): os.kill(event['pid'], 0)
                self.assertIn(('compiler stdout ' + event['key']).encode(), lines)
                self.assertIn(('compiler stderr ' + event['key']).encode(), stderr.splitlines())
            for path, data in inputs.items(): self.assertEqual(path.read_bytes(), data)
            if accepted and non_c:
                for group, count in enumerate((ordinary, shared)):
                    for index in range(count):
                        self.assertEqual((work / f'{stem}_{group}_{index}.s').read_bytes(),
                                         inputs[work / f'__snapshot_{group}_{index}.i'])

    def test_overlap_bounds_tail_and_both_output_phases(self):
        for limit in (1, 2, 4, 8):
            for stem in ('__snapshot', '__assembly'):
                with self.subTest(limit=limit, stem=stem): self.run_case(limit=limit, stem=stem)

    def test_exit_signal_spawn_and_interrupted_wait(self):
        for options in ({'failure': True}, {'signaled': True}, {'spawn': 0},
                        {'spawn': 1}, {'interrupt': True}):
            with self.subTest(options=options): self.run_case(**options)

    def test_non_c_copy_empty_shared_only_and_invalid_limits(self):
        for options in ({'non_c': True}, {'ordinary': 0, 'shared': 0},
                        {'ordinary': 0, 'shared': 3}, {'limit': 0}, {'limit': 9}):
            with self.subTest(options=options): self.run_case(**options)


if __name__ == '__main__': unittest.main()
