"""I reject late clean exits even when my parent misses its polling wakeup."""
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
LATE_ENTRY = r'''
#define _POSIX_C_SOURCE 200809L
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
int64_t delayed_finish(void) {
    signal(SIGALRM, SIG_IGN);
    fprintf(stderr, "REAP_CONTROL_READY %ld\n", (long)getpid());
    fflush(stderr);
    const char *gate = getenv("NANO_REAP_GATE");
    while (gate && access(gate, F_OK) != 0) {
        struct timespec check = {0, 1000000};
        nanosleep(&check, NULL);
    }
    struct timespec pause = {1, 200000000};
    while (nanosleep(&pause, &pause) < 0) {}
    struct timespec finished;
    if (clock_gettime(CLOCK_MONOTONIC, &finished) != 0) return -1;
    fprintf(stderr, "REAP_CONTROL_FINISH %ld %ld %ld\n", (long)getpid(),
            (long)finished.tv_sec, (long)finished.tv_nsec);
    fflush(stderr);
    return 42;
}
'''


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'),
                     'I require Linux process sleep and exit observations.')
class ShadowReapDeadline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'.test_output').mkdir(exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix='shadow-reap-', dir=ROOT/'.test_output'))
        print('I retain actual delayed-parent controls at', cls.work, flush=True)

    def delayed_parent(self, argv, directory):
        gate = directory/'release-gate'
        observations = {}
        environment = dict(os.environ, TMPDIR=str(directory), NANO_REAP_GATE=str(gate),
                           NANO_SHADOW_TIMEOUT_SECONDS='1',
                           NANO_SHADOW_PHASE_TRACE='1', NANO_SHADOW_TRACE='1')
        stopped = False
        output = (directory/'stdout').open('wb')
        process = subprocess.Popen(list(map(str, argv)), cwd=ROOT, env=environment,
                                   stdout=output, stderr=subprocess.PIPE, start_new_session=True)
        error = b''
        try:
            poller = select.poll()
            poller.register(process.stderr, select.POLLIN | select.POLLHUP)
            ready_deadline = time.monotonic()+20
            child = None
            while child is None and time.monotonic() < ready_deadline:
                if not poller.poll(100):
                    continue
                chunk = os.read(process.stderr.fileno(), 65536)
                if not chunk:
                    break
                error += chunk
                for line in error.splitlines():
                    if line.startswith(b'REAP_CONTROL_READY '):
                        child = int(line.split()[1])
            self.assertIsNotNone(child, error.decode(errors='replace'))
            # I stop only a parent actually sleeping between its clock and next waitpid.
            sleep_deadline = time.monotonic()+0.2
            sleeping = False
            while time.monotonic() < sleep_deadline:
                sleeping = Path(f'/proc/{process.pid}/wchan').read_text() == 'hrtimer_nanosleep'
                if sleeping:
                    break
                os.sched_yield()
            self.assertTrue(sleeping, 'I did not observe the original parent polling sleep.')
            os.kill(process.pid, signal.SIGSTOP)
            stopped = True
            stop_deadline = time.monotonic()+0.2
            while time.monotonic() < stop_deadline:
                state = Path(f'/proc/{process.pid}/status').read_text()
                if '\nState:\tT (stopped)\n' in '\n'+state:
                    break
                os.sched_yield()
            self.assertIn('\nState:\tT (stopped)\n', '\n'+state)
            observations['parent_stopped_monotonic'] = time.monotonic()
            observations['child_pid'] = child
            gate.write_text('go\n')
            descriptor = os.pidfd_open(child)
            try:
                exited = select.poll()
                exited.register(descriptor, select.POLLIN)
                self.assertTrue(exited.poll(3000), 'My actual shadow child did not exit.')
            finally:
                os.close(descriptor)
            os.kill(process.pid, signal.SIGCONT)
            stopped = False
            error += process.communicate(timeout=20)[1]
            finished = [line.split() for line in error.splitlines()
                        if line.startswith(b'REAP_CONTROL_FINISH ')]
            self.assertEqual(len(finished), 1, error.decode(errors='replace'))
            self.assertEqual(int(finished[0][1]), child)
            stamp = int(finished[0][2])+int(finished[0][3])/1e9
            observations['child_finished_monotonic'] = stamp
            self.assertGreater(stamp-observations['parent_stopped_monotonic'], 1)
            self.assertNotEqual(process.returncode, 0, error.decode(errors='replace'))
            self.assertIn(b'after 1 seconds', error)
            return error
        finally:
            if stopped:
                os.kill(process.pid, signal.SIGCONT)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                error += process.communicate(timeout=5)[1]
            output.close()
            (directory/'stderr').write_bytes(error)
            (directory/'process.json').write_text(json.dumps(dict(argv=list(map(str, argv)),
                returncode=process.returncode, deadline_seconds=1, observations=observations))+'\n')

    def source_case(self, label, compiler, *options):
        directory = self.work/label
        directory.mkdir()
        foreign = directory/'foreign'
        foreign.mkdir()
        (foreign/'module.json').write_text(json.dumps(dict(name='shadow_reap', c_sources=['delay.c'])))
        (foreign/'delay.c').write_text(LATE_ENTRY)
        (foreign/'delay.nano').write_text('pub extern fn delayed_finish() -> int\n')
        source = directory/'input.nano'
        source.write_text(f'module "{foreign / "delay.nano"}" as delay\n'
                          'fn main() -> int { return 0 }\n'
                          'shadow main { unsafe { assert (== (delay.delayed_finish) 42) } }\n')
        product = directory/'product'
        prior = b'I retain the prior output after late completion.\n'
        product.write_bytes(prior)
        before = hashlib.sha256(compiler.read_bytes()).hexdigest()
        error = self.delayed_parent([compiler, source, *options, '-o', product], directory)
        self.assertEqual(product.read_bytes(), prior)
        self.assertEqual(hashlib.sha256(compiler.read_bytes()).hexdigest(), before)
        return error

    def test_nanovirt_preserves_output_after_late_clean_exit(self):
        self.source_case('nanovirt', ROOT/'bin/nano_virt', '--emit-nvm')

    def test_cseed_preserves_output_after_late_clean_exit(self):
        error = self.source_case('cseed', ROOT/'bin/nanoc_c')
        events = [json.loads(line) for line in error.splitlines()
                  if line.startswith(b'{"event":"cseed-shadow-phase"')]
        status = events[-1]
        self.assertTrue(status['reaped'] and status['completed'] and status['parent_timeout'])
        self.assertFalse(status['clock_failure'])
        self.assertEqual((status['exit_code'], status['signal']), (0, 0))
        complete = next(e for e in events if e['phase']=='completion-written')
        elapsed = (complete['monotonic_seconds']-status['parent_start_seconds'] +
                   (complete['monotonic_nanoseconds']-status['parent_start_nanoseconds'])/1e9)
        self.assertGreater(elapsed, 1)

    def test_runtime_rejects_late_clean_exit(self):
        directory = self.work/'runtime'
        directory.mkdir()
        source = directory/'control.c'
        source.write_text(LATE_ENTRY+'\n#include "runtime/shadow_runner.h"\n'
                          'static int entry(void) { return delayed_finish()==42 ? 0 : 1; }\n'
                          'int main(void) { return nl_run_shadow_entry(entry, 1); }\n')
        binary = directory/'control'
        command = [*shlex.split(os.environ.get('CC', 'cc')), '-std=c99', '-Wall', '-Wextra',
                   '-Werror', '-O1', '-g', '-Isrc', str(source), '-o', str(binary)]
        built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
        (directory/'build.stdout').write_bytes(built.stdout)
        (directory/'build.stderr').write_bytes(built.stderr)
        self.assertEqual(built.returncode, 0, built.stderr)
        self.delayed_parent([binary], directory)


if __name__ == '__main__':
    unittest.main()
