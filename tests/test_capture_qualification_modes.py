"""I keep complete private Clang observations separate from GNU admission."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import ci_qualify_as246 as qualifier


class CaptureQualificationModes(unittest.TestCase):
    @unittest.skipUnless(hasattr(__import__('os'), 'killpg'), 'I require native process groups.')
    def test_controller_kill_keeps_already_written_command_output(self):
        import os
        import signal
        import subprocess
        import time
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            pidfile = out / 'child.pid'
            child = ('import os,pathlib,sys,time; '
                     'print("retained-before-kill",flush=True); '
                     'print("retained-error-before-kill",file=sys.stderr,flush=True); '
                     f'pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(30)')
            parent = (f'from pathlib import Path; from scripts import ci_qualify_as246 as q; '
                      f'q.OUT=Path({str(out)!r}); q.command({[sys.executable, '-c', child]!r},"killed",30)')
            controller = subprocess.Popen([sys.executable, '-c', parent], cwd=qualifier.ROOT,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            child_pid = None
            try:
                end = time.monotonic() + 5
                while time.monotonic() < end and not pidfile.exists():
                    time.sleep(0.02)
                self.assertTrue(pidfile.exists(), 'I require a live child before controller loss.')
                child_pid = int(pidfile.read_text())
                time.sleep(0.05)
                controller.kill()
                controller.wait(timeout=5)
                self.assertEqual((out / 'killed.stdout').read_bytes(), b'retained-before-kill\n')
                self.assertEqual((out / 'killed.stderr').read_bytes(), b'retained-error-before-kill\n')
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_timed_out_command_retains_both_partial_streams(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            child = [sys.executable, '-c',
                     'import sys,time; print("retained-out",flush=True); '
                     'print("retained-err",file=sys.stderr,flush=True); time.sleep(30)']
            with patch.object(qualifier, 'ROOT', output), patch.object(qualifier, 'OUT', output):
                with self.assertRaises(subprocess.TimeoutExpired):
                    qualifier.command(child, 'deadline', 1)
            self.assertEqual((output / 'deadline.stdout').read_bytes(), b'retained-out\n')
            self.assertEqual((output / 'deadline.stderr').read_bytes(), b'retained-err\n')

    def fixture(self, directory):
        root = Path(directory)
        source = root / 'src/module_builder.c'
        test = root / 'tests/test_source_snapshots.py'
        source.parent.mkdir()
        test.parent.mkdir()
        # I use the actual source fixtures, not a second copy of their tables.
        source.write_bytes((qualifier.ROOT / source.relative_to(root)).read_bytes())
        test.write_bytes((qualifier.ROOT / test.relative_to(root)).read_bytes())
        observation = root / 'observation'
        observation.mkdir()
        (observation / 'observation.json').write_text(json.dumps({
            'assembler_version': 'GNU assembler (fixture) 2.46\n', 'commit': 'fixture'}))
        return root, source, test, observation

    def test_complete_candidate_refuses_different_selected_cc_and_restores_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, test, observation = self.fixture(directory)
            originals = {p: p.read_bytes() for p in (source, test)}
            output = root / 'result'
            clang, cc = root / 'clang', root / 'cc'
            for path in (clang, cc):
                path.write_text('#!' + sys.executable + '\nprint("Ubuntu clang version 21.1.8 (6ubuntu1)")\n')
                path.chmod(0o700)
            def selected(name):
                return str(clang if name == 'clang' else cc)
            with patch.object(qualifier, 'ROOT', root), patch.object(qualifier, 'OUT', output), \
                 patch.object(qualifier.capture, 'OUT', observation), \
                 patch.object(qualifier.shutil, 'which', side_effect=selected), \
                 patch.object(sys, 'argv', ['qualify', '--clang-2118-complete']):
                with self.assertRaisesRegex(RuntimeError, 'same actual qualification driver'):
                    qualifier.main()
            self.assertTrue(all(p.read_bytes() == raw for p, raw in originals.items()))
            result = json.loads((output / 'result.json').read_text())
            self.assertTrue(result['checkout_sources_restored'])
            self.assertFalse(result['product_admission_changed'])
            self.assertFalse((output / 'candidate.patch').exists())
            self.assertFalse((output / 'build.stdout').exists())

    def test_complete_candidate_refuses_unqualified_actual_clang_and_restores_inputs(self):
        clang = shutil.which('clang')
        if not clang:
            self.skipTest('I require an actual installed Clang for this refusal control.')
        import subprocess
        version = subprocess.check_output([clang, '--version'], text=True, timeout=10)
        if version.startswith('Ubuntu clang version 21.1.8 (6ubuntu1)\n'):
            self.skipTest('I exercise a different actual Clang identity here.')
        with tempfile.TemporaryDirectory() as directory:
            root, source, test, observation = self.fixture(directory)
            originals = {p: p.read_bytes() for p in (source, test)}
            output = root / 'result'
            with patch.object(qualifier, 'ROOT', root), patch.object(qualifier, 'OUT', output), \
                 patch.object(qualifier.capture, 'OUT', observation), \
                 patch.object(sys, 'argv', ['qualify', '--clang-2118-complete']):
                with self.assertRaisesRegex(RuntimeError, 'unqualified selected Clang identity'):
                    qualifier.main()
            self.assertTrue(all(p.read_bytes() == raw for p, raw in originals.items()))
            result = json.loads((output / 'result.json').read_text())
            self.assertTrue(result['checkout_sources_restored'])
            self.assertFalse((output / 'candidate.patch').exists())
            self.assertFalse((output / 'build.stdout').exists())


class FixedAssemblerAdmission(unittest.TestCase):
    def test_identified_native_predicate_preserves_generic_refusals(self):
        import ast
        import hashlib
        import os
        import subprocess
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('I require a native C compiler for the generated predicate.')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / 'backend'
            (backend / 'bin').mkdir(parents=True)
            image = backend / 'bin/as'
            image.write_bytes(b'constructed selection fixture; I never execute this image')
            banner = 'GNU assembler (GNU Binutils for NanoLang retained capture r1) 2.46.0.20260210\n'
            compiler_path = str(Path(compiler).resolve())
            previous = {'compiler': compiler_path,
                'tool_sha256': {compiler_path: hashlib.sha256(Path(compiler_path).read_bytes()).hexdigest()}}
            metadata = {'completed': True, 'source_sha256': 'd75a94f4d73e7a4086f7513e67e439e8fcdcbb726ffe63f4661744e6256b2cf2',
                'fixed_source_sha256': '6ffe0ed82d02c8f92198b4c2b5200a8093b7eea0a07a96bf951eecabf021b036',
                'assembler_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                'assembler_version': banner, 'gas_expected_passes': 1239,
                'witnesses': [{'original_source_identity': True, 'expected': True}] * 8}
            (backend / 'build.json').write_text(json.dumps(metadata))
            def query(argv, label, timeout):
                value = str(image) + '\n' if label == 'fixed-selected-assembler' else banner
                (root / (label + '.stdout')).write_text(value)
            original_source = (qualifier.ROOT / 'src/module_builder.c').read_text()
            original_test = (qualifier.ROOT / 'tests/test_source_snapshots.py').read_text()
            with patch.dict(os.environ, NANO_FIXED_AS246_DIRECTORY=str(backend)), \
                 patch.object(qualifier, 'OUT', root), patch.object(qualifier, 'command', side_effect=query):
                source, test, _ = qualifier.fixed_assembler_candidate(original_source, original_test, previous)
            ast.parse(test)
            self.assertIn('static const char *versions[] = {" 2.40", " 2.42"};', source)
            start = source.index('static bool module_assembler_version_supported(')
            end = source.index('\n#endif', start)
            native = root / 'predicate'
            program = ('#include <stdbool.h>\n#include <stddef.h>\n#include <string.h>\n' +
                source[start:end] + '\nint main(int n,char **v){return !module_assembler_version_supported(n>1?v[1]:NULL); }\n')
            built = subprocess.run([compiler, '-std=c99', '-Wall', '-Wextra', '-Werror', '-x', 'c',
                '-o', str(native), '-'], input=program.encode(), capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            for value, accepted in [(banner, True), (banner.replace('capture r1', 'capture r2'), False),
                    ('GNU assembler (GNU Binutils) 2.46.0.20260210\n', False),
                    ('GNU assembler (GNU Binutils for Ubuntu) 2.46\n', False),
                    ('GNU assembler (GNU Binutils) 2.42\n', True), (banner.rstrip('\n'), False)]:
                with self.subTest(version=value):
                    run = subprocess.run([str(native), value], capture_output=True, timeout=5)
                    self.assertEqual(run.returncode, 0 if accepted else 1, run.stderr)
            self.assertEqual(subprocess.run([str(native)], timeout=5).returncode, 1)


class QualifiedAssemblerSource(unittest.TestCase):
    def test_drifted_include_source_is_refused_without_mutation(self):
        from scripts import ci_build_capture_assembler as builder
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input-scrub.c'
            source.write_bytes(b'unexpected upstream source')
            with self.assertRaisesRegex(RuntimeError, 'unexpected upstream include-state source'):
                builder.apply_include_fix(source)
            self.assertEqual(source.read_bytes(), b'unexpected upstream source')

    def test_changed_download_is_preserved_and_refused_before_extraction_or_build(self):
        import io
        from scripts import ci_build_capture_assembler as builder
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            with patch.object(builder, 'OUT', output), \
                 patch.object(builder.urllib.request, 'urlopen', return_value=io.BytesIO(b'changed source')), \
                 patch.object(builder, 'command') as command:
                with self.assertRaisesRegex(RuntimeError, 'upstream release checksum'):
                    builder.main()
            command.assert_not_called()
            self.assertEqual((output / 'binutils-2.42.tar.xz').read_bytes(), b'changed source')
            self.assertFalse((output / 'bin/as').exists())
            record = json.loads((output / 'build.json').read_text())
            self.assertNotIn('completed', record)
            self.assertIn('checksum', record['error'])


if __name__ == '__main__':
    unittest.main()
