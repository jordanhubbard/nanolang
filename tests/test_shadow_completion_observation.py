"""I observe actual C-seed body/report/exit boundaries without changing admission."""
import hashlib
import copy
import json
import os
from pathlib import Path
import signal
import shlex
import sys
import tempfile
import unittest

from tests.native_process import run_captured
from scripts.compare_cseed_source_creation import shadow_phases

ROOT = Path(__file__).resolve().parents[1]


class ShadowCompletionObservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='shadow-completion-', dir=ROOT/'.test_output'))
        cls.compiler = ROOT/'bin/nanoc_c'
        cls.compiler_hash = hashlib.sha256(cls.compiler.read_bytes()).hexdigest()
        cls.operations = []
        print('I retain actual shadow-completion controls at', cls.work, flush=True)

    @classmethod
    def tearDownClass(cls):
        assert hashlib.sha256(cls.compiler.read_bytes()).hexdigest() == cls.compiler_hash
        (cls.work/'provider-before-after.json').write_text(json.dumps({str(cls.compiler): cls.compiler_hash})+'\n')

    def compile(self, label, body='assert true', *, observe=True, report=None, declaration='', mutate_report=None):
        directory = self.work/label; directory.mkdir()
        source, product = directory/'input.nano', directory/'program'
        if mutate_report:
            operation = 'os.mkdir' if mutate_report == 'directory' else 'os.mkfifo'
            script = 'import os;os.unlink('+repr(str(report))+');'+operation+'('+repr(str(report))+')'
            command = shlex.join([sys.executable, '-c', script])
            declaration += '\nextern fn system(command: string) -> int'
            body = 'unsafe { assert (== (system '+json.dumps(command)+') 0) } '+body
        source.write_text(declaration+'\nfn main() -> int { return 0 }\nshadow main { '+body+' }\n')
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        product.write_bytes(b'I retain the prior output.\n')
        environment = dict(os.environ, NANO_SHADOW_TRACE='1', NANO_SHADOW_TIMEOUT_SECONDS='1')
        environment.pop('NANO_SHADOW_PHASE_TRACE', None)
        environment.pop('NANO_LLM_SHADOW_JSON', None)
        if observe: environment['NANO_SHADOW_PHASE_TRACE'] = '1'
        argv = [str(self.compiler), str(source), '-o', str(product)]
        if report: argv += ['--llm-shadow-json', str(report)]
        result = run_captured(argv, cwd=ROOT, env=environment, timeout=15)
        (directory/'stdout').write_bytes(result.stdout); (directory/'stderr').write_bytes(result.stderr)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        events = [json.loads(line) for line in result.stderr.decode().splitlines()
                  if line.startswith('{"event":"cseed-shadow-phase"')]
        bodies = [json.loads(line) for line in result.stderr.decode().splitlines()
                  if line.startswith('{"event":"shadow-')]
        if result.returncode:
            self.assertEqual(product.read_bytes(), b'I retain the prior output.\n')
        else:
            data = product.read_bytes()
            self.assertIn(data[:4], (b'\xcf\xfa\xed\xfe', b'\x7fELF'))
            execution = run_captured([product], cwd=ROOT, timeout=5)
            self.assertEqual(execution.returncode, 0, execution.stderr)
        self.operations.append({'label': label, 'argv': argv, 'returncode': result.returncode,
                                'input_sha256_before_after': before, 'observation': observe,
                                'phases': events, 'bodies': bodies})
        (self.work/'operations.json').write_text(json.dumps(self.operations, indent=2)+'\n')
        return result, events, bodies

    def status(self, events):
        self.assertTrue(events)
        self.assertTrue(all(e['clock_ok'] for e in events))
        stamps = [e['monotonic_seconds']*10**9+e['monotonic_nanoseconds'] for e in events]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(events[-1]['phase'], 'supervisor-status')
        status = events[-1]
        self.assertTrue(status['reaped'])
        self.assertFalse(status['clock_failure'])
        self.assertNotEqual(status['pid'], status['child_pid'])
        self.assertTrue(all(e['pid'] == status['child_pid'] for e in events[:-1]))
        self.assertEqual(status['deadline_seconds'], 1)
        return status

    def test_success_and_default_output(self):
        plain, events, bodies = self.compile('disabled', observe=False)
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertEqual(events, [])
        traced, events, traced_bodies = self.compile('enabled')
        self.assertEqual(traced.returncode, 0, traced.stderr)
        self.assertEqual(plain.stdout, traced.stdout)
        self.assertEqual([b['event'] for b in bodies], [b['event'] for b in traced_bodies])
        self.assertEqual([e['phase'] for e in events], ['dispatch-start', 'report-start', 'report-returned',
                         'dispatch-returned', 'completion-written', 'flush-start', 'flush-end', 'supervisor-status'])
        status = self.status(events)
        self.assertTrue(status['completed'])
        self.assertFalse(status['parent_timeout'])
        self.assertEqual((status['signal'], status['exit_code']), (0, 0))

    def test_assertion_and_failed_report_preserve_output(self):
        result, events, bodies = self.compile('assertion', 'assert false')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(bodies[-1]['failures'], 1)
        status = self.status(events)
        self.assertFalse(status['completed'])
        self.assertEqual((status['signal'], status['exit_code']), (0, 1))
        result, events, bodies = self.compile('failed-report', report=self.work/'failed-report.json', mutate_report='directory')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(bodies[-1]['event'], 'shadow-complete')
        self.assertEqual(next(e['value'] for e in events if e['phase']=='report-returned'), 0)
        self.assertFalse(self.status(events)['completed'])

    def test_body_deadline(self):
        result, events, bodies = self.compile('body-timeout', 'while true {}')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'after 1 seconds', result.stderr)
        self.assertEqual([b['event'] for b in bodies], ['shadow-start'])
        self.assertEqual([e['phase'] for e in events], ['dispatch-start', 'supervisor-status'])
        status = self.status(events)
        self.assertIn(status['signal'], (signal.SIGALRM, signal.SIGKILL))
        self.assertFalse(status['completed'])

    def test_blocked_report_after_completed_body(self):
        report = self.work/'blocked-report.fifo'
        result, events, bodies = self.compile('report-timeout', report=report, mutate_report='fifo')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'after 1 seconds', result.stderr)
        self.assertEqual([b['event'] for b in bodies], ['shadow-start', 'shadow-complete'])
        self.assertEqual([e['phase'] for e in events], ['dispatch-start', 'report-start', 'supervisor-status'])
        status = self.status(events)
        self.assertIn(status['signal'], (signal.SIGALRM, signal.SIGKILL))
        self.assertFalse(status['completed'])

    def test_early_zero_exit_is_not_completion(self):
        result, events, _ = self.compile('early-exit', 'unsafe { (exit 0) }',
                                        declaration='extern fn exit(code: int) -> void')
        self.assertNotEqual(result.returncode, 0)
        status = self.status(events)
        self.assertEqual((status['signal'], status['exit_code']), (0, 0))
        self.assertFalse(status['completed'])

    def test_native_observer_preserves_errno_on_io_failure(self):
        source, binary = self.work/'errno.c', self.work/'errno-native'
        source.write_text('''#define _POSIX_C_SOURCE 200809L
#include "runtime/shadow_observation.h"
#include <assert.h>
int main(void) {
    assert(close(STDERR_FILENO) == 0);
    errno = E2BIG;
    nl_shadow_observe_phase(false, "disabled", 0);
    assert(errno == E2BIG && !ferror(stderr));
    errno = ERANGE;
    nl_shadow_observe_phase(true, "failed-write", 0);
    assert(errno == ERANGE && ferror(stderr));
    errno = ENOTTY;
    nl_shadow_observe_status(true, getpid(), getpid(), 0, false, false, true, 1, NULL);
    assert(errno == ENOTTY);
    puts("I preserve errno and leave disabled output untouched even with a closed diagnostic descriptor.");
    return 0;
}
''')
        cc = shlex.split(os.environ.get('CC', 'cc'))
        argv = [*cc, '-std=c99', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-UNDEBUG',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-Isrc', str(source), '-o', str(binary)]
        built = run_captured(argv, cwd=ROOT, timeout=60)
        self.assertEqual(built.returncode, 0, built.stderr)
        before = hashlib.sha256(binary.read_bytes()).hexdigest()
        result = run_captured([binary], cwd=ROOT, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b'I preserve errno', result.stdout)
        self.assertEqual(hashlib.sha256(binary.read_bytes()).hexdigest(), before)
        self.operations.append({'label': 'native-errno', 'argv': argv, 'build_returncode': built.returncode,
                                'returncode': result.returncode, 'native_sha256_before_after': before})
        (self.work/'operations.json').write_text(json.dumps(self.operations, indent=2)+'\n')

    def test_phase_audit_refuses_tampered_actual_records(self):
        result, events, _ = self.compile('phase-audit')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(shadow_phases(result.stderr.decode(), True, 1)['valid'])
        for field, value in (('child_pid', events[-1]['child_pid']+1),
                             ('wait_status', 7 << 8), ('completed', 'true'),
                             ('parent_timeout', True), ('deadline_seconds', 2),
                             ('parent_start_clock_ok', False)):
            altered = copy.deepcopy(events); altered[-1][field] = value
            text = '\n'.join(json.dumps(e, separators=(',', ':')) for e in altered)
            self.assertFalse(shadow_phases(text, True, 1)['valid'], field)
        partial = result.stderr.decode()+'\n{"event":"cseed-shadow-phase","phase":'
        self.assertFalse(shadow_phases(partial, True, 1)['valid'])


if __name__ == '__main__':
    unittest.main()
