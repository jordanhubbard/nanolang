"""I exercise timeout observation with real child processes and retained pipes."""
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import unittest

from scripts import ci_stage2_emitter_diagnostic as diagnostic


@unittest.skipUnless(sys.platform == 'linux', 'I sample Linux process groups here.')
class Stage2EmitterDiagnostic(unittest.TestCase):
    def test_sampler_identity_requires_the_real_elf_tool(self):
        from scripts.stage2_cpu_sampler import Sampler
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = Path(temporary)/'perf'
            wrapper.write_text('#!/bin/sh\nexit 0\n')
            with self.assertRaises(AssertionError):
                Sampler(perf=wrapper)

    def test_cpu_sampler_finishes_after_the_observed_group_is_reaped(self):
        owner = self
        class LifecycleProbe:
            def start(self, pid, directory):
                owner.assertEqual(os.getpgid(pid), pid)
                self.pid = pid
                self.directory = directory
                return pid
            def finish(self, pid):
                owner.assertEqual(pid, self.pid)
                with owner.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
                return {'protocol_probe_only': True}
        with tempfile.TemporaryDirectory() as temporary:
            probe = LifecycleProbe()
            row = diagnostic.observe([sys.executable, '-c', 'import signal; signal.pause()'],
                                     Path(temporary)/'command', cwd=temporary,
                                     deadline=0.1, interval=0.01, cpu_sampler=probe)
            self.assertTrue(row['command_timeout'])
            self.assertTrue(row['cpu_profile']['protocol_probe_only'])
            self.assertEqual(probe.directory, Path(temporary)/'command/cpu-profile')

    def test_failed_sampler_admission_reaps_the_product(self):
        class RefusedSampler:
            def start(self, pid, directory):
                self.pid = pid
                raise RuntimeError('I refuse this diagnostic sampler.')
        with tempfile.TemporaryDirectory() as temporary:
            probe = RefusedSampler()
            with self.assertRaisesRegex(RuntimeError, 'refuse this diagnostic sampler'):
                diagnostic.observe([sys.executable, '-c', 'import signal; signal.pause()'],
                                   Path(temporary)/'command', cwd=temporary,
                                   deadline=1, cpu_sampler=probe)
            with self.assertRaises(ProcessLookupError):
                os.kill(probe.pid, 0)
            row = json.loads((Path(temporary)/'command/result.json').read_text())
            self.assertTrue(row['terminal'] and row['sampler_stopped'])

    def test_native_process_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = diagnostic.controls(root / 'controls')
            self.assertEqual(len(rows), 3)
            self.assertEqual((root / 'controls/complete/stdout').read_bytes(), b'complete\n')
            self.assertEqual((root / 'controls/nonzero/stderr').read_bytes(), b'refused\n')
            sampled = [json.loads(line) for line in
                       (root / 'controls/timeout/processes.jsonl').read_text().splitlines()]
            self.assertTrue(any(p['pid'] == rows[2]['pid']
                                for row in sampled for p in row['processes']))
            self.assertTrue(all(p['group'] == rows[2]['pid']
                                for row in sampled for p in row['processes']))

    def test_observer_retains_timeout_stderr_and_exact_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = 'import signal,sys; print("before-timeout",file=sys.stderr,flush=True); signal.pause()'
            row = diagnostic.observe([sys.executable, '-c', source], root / 'command',
                                     cwd=root, deadline=0.2, interval=0.02)
            self.assertTrue(row['command_timeout'])
            self.assertEqual(row['exit_code'], -9)
            self.assertEqual((root / 'command/stderr').read_bytes(), b'before-timeout\n')
            self.assertEqual(row['stderr_sha256'], diagnostic.digest(root / 'command/stderr'))
            self.assertTrue(row['terminal'] and row['sampler_stopped'])
            self.assertFalse(row['observer_errors'])

    def test_original_command_requires_exact_product_and_private_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, work = base / 'product', base / 'temporary'
            command = [str(root / 'bin/nanoc_stage2'),
                       str(root / 'tests/selfhost_shadow_emitter.nano'), '-o',
                       str(work / 'nano-qualification-original/nano-native-shadows-original/nanoc_stage2')]
            log = f"subprocess.TimeoutExpired: Command '{command!r}' timed out after 300 seconds\n"
            self.assertEqual(diagnostic.original_command(root, log, work), command)
            for old, new in [('300 seconds', '301 seconds'), ('bin/nanoc_stage2', 'bin/nanoc_c'),
                             ('nano-native-shadows-original', 'unowned'),
                             (str(work), str(base / 'outside'))]:
                with self.subTest(replacement=new), self.assertRaises(AssertionError):
                    diagnostic.original_command(root, log.replace(old, new), work)
            Path(command[-1]).parent.mkdir(parents=True)
            with self.assertRaises(AssertionError):
                diagnostic.original_command(root, log, work)

    def test_partial_compiler_files_survive_without_following_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'work'
            work.mkdir()
            (work / 'partial.c').write_bytes(b'int partial;\n')
            outside = root / 'outside'
            outside.mkdir()
            (outside / 'unrelated').write_text('outside')
            (work / 'link').symlink_to(outside, target_is_directory=True)
            rows = diagnostic.retain_temporaries(work, root / 'retained')
            self.assertEqual((root / 'retained/tree/partial.c').read_bytes(), b'int partial;\n')
            self.assertFalse((root / 'retained/tree/link').exists())
            self.assertEqual(next(row['link'] for row in rows if row['path'] == 'link'), str(outside))
            with self.assertRaises(AssertionError):
                diagnostic.retain_temporaries(work, work / 'recursive')

    def test_restored_bytes_also_admit_the_next_atomic_publication(self):
        class RetainedArtifact:
            def restore(self, root, out):
                self.cache = root / 'modules/nanoisa/.build'
                for name in ['current', '.nano-gen-AbC123']:
                    path = self.cache / name
                    path.mkdir(parents=True)
                    (path / 'library').write_bytes(b'original')

            def restore_cache_currents(self, root, out):
                (self.cache / 'current').rename(out / 'flattened')
                (self.cache / 'current').symlink_to('.nano-gen-AbC123')
                report = {'complete': True, 'entries': [
                    {'current': 'modules/nanoisa/.build/current', 'target': '.nano-gen-AbC123'}]}
                diagnostic.save(out / 'cache-current-restoration.json', report)
                return report

            def product_tree(self, root, out):
                self.asserted_tree = True

            def identities(self, root):
                return {'library': diagnostic.digest(self.cache / 'current/library')}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / 'evidence'
            out.mkdir()
            helper = RetainedArtifact()
            before = diagnostic.restore_admitted_inputs(helper, root, out)
            self.assertEqual(before['library'], diagnostic.digest(out / 'flattened/library'))
            next_generation = helper.cache / '.nano-gen-NeXt12'
            next_generation.mkdir()
            (next_generation / 'library').write_bytes(b'next')
            pointer = helper.cache / 'next'
            pointer.symlink_to(next_generation.name)
            os.replace(pointer, helper.cache / 'current')
            self.assertEqual((helper.cache / 'current/library').read_bytes(), b'next')
            self.assertEqual((out / 'flattened/library').read_bytes(), b'original')

    def test_native_readiness_runs_the_elf_and_rejects_its_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            for tool in ['true', 'false']:
                root = Path(temporary) / tool
                (root / 'bin').mkdir(parents=True)
                out = root / 'evidence'
                out.mkdir()
                compiler = root / 'bin/nanoc_stage2'
                compiler.write_text('#!/bin/sh\ncp ' + shlex.quote(shutil.which(tool)) + ' "$3"\n')
                compiler.chmod(0o700)
                if tool == 'true':
                    compiled, executed = diagnostic.native_control(root, out)
                    self.assertEqual((compiled['exit_code'], executed['exit_code']), (0, 0))
                else:
                    with self.assertRaises(AssertionError):
                        diagnostic.native_control(root, out)
                    self.assertEqual(json.loads((out / 'native-control/execute/result.json').read_text())['exit_code'], 1)


if __name__ == '__main__':
    unittest.main()
