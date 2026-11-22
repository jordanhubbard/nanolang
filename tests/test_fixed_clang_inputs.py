"""I refuse incomplete producers and drift before a complete Clang consumer."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import ci_fixed_clang_inputs as inputs
from tests import test_fixed_userguide_qualification as guide_tests
from tests import test_capture_qualification_modes as mode_tests


class FixedClangInputs(unittest.TestCase):
    def test_fixed_clang_requires_the_same_actual_driver_and_restores_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, test, observation = mode_tests.CaptureQualificationModes().fixture(directory)
            originals = {path: path.read_bytes() for path in (source, test)}
            clang, cc = root / 'clang', root / 'cc'
            for path in (clang, cc):
                path.write_text('#!' + sys.executable + '\nprint("Ubuntu clang version 21.1.8 (6ubuntu1)")\n')
                path.chmod(0o700)
            output = root / 'result'
            q = inputs.qualifier
            with patch.object(q, 'ROOT', root), patch.object(q, 'OUT', output), \
                 patch.object(q.capture, 'OUT', observation), \
                 patch.object(q.shutil, 'which', side_effect=lambda name: str(clang if name == 'clang' else cc)), \
                 patch.object(q, 'fixed_assembler_candidate', side_effect=lambda s, t, p: (s, t, {})), \
                 patch.object(sys, 'argv', ['qualify', '--fixed-clang-complete']):
                with self.assertRaisesRegex(RuntimeError, 'same actual qualification driver'):
                    q.main()
            self.assertTrue(all(path.read_bytes() == raw for path, raw in originals.items()))
            result = json.loads((output / 'result.json').read_text())
            self.assertTrue(result['fixed_clang_complete'])
            self.assertFalse(result['fixed_as246_complete'])
            self.assertTrue(result['checkout_sources_restored'])
            self.assertFalse((output / 'candidate.patch').exists())

    def test_incomplete_producer_cannot_authorize_a_new_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory)
            candidate = prior / 'as246-candidate'
            candidate.mkdir()
            result = guide_tests.FixedUserguideQualification().fixture(candidate)
            result['source_pin'] = 'a' * 40
            (candidate / 'result.json').write_text(json.dumps(result))
            run = {'id': 1, 'head_sha': 'b' * 40, 'status': 'completed', 'conclusion': 'success',
                   'path': '.github/workflows/source-capture-toolchain-diagnostic.yml'}
            with self.assertRaisesRegex(RuntimeError, 'matching source-capture producer'):
                inputs.require_prior(prior, run)
            run['head_sha'] = result['source_pin']
            with self.assertRaisesRegex(RuntimeError, 'ordinary default guide'):
                inputs.require_prior(prior, run)

    def test_changed_consumer_source_or_patch_is_refused_without_writing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / 'prior'
            prior.mkdir()
            source, test = root / 'module_builder.c', root / 'test_source_snapshots.py'
            source.write_bytes(b'qualified C source\n')
            test.write_bytes(b'qualified original test fixture\n')
            for path in (source, test):
                (prior / path.name).write_bytes(path.read_bytes())
            (prior / 'candidate.patch').write_bytes(b'qualified patch\n')
            result = inputs.require_candidate(prior, source, test, b'qualified patch\n')
            self.assertEqual(result['source_sha256'], inputs.digest(source))
            with self.assertRaisesRegex(RuntimeError, 'exact previously qualified consumer patch'):
                inputs.require_candidate(prior, source, test, b'different patch\n')
            source.write_bytes(b'drifted C source\n')
            with self.assertRaisesRegex(RuntimeError, 'byte-identical previously qualified'):
                inputs.require_candidate(prior, source, test, b'qualified patch\n')
            self.assertEqual(source.read_bytes(), b'drifted C source\n')

    def test_real_git_continuity_refuses_assertion_drift_after_package_fixture_upgrade(self):
        from tests.assembler_search_fixture import upgrade_source
        from tests.retained_capture_evidence import pre_admission_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tests').mkdir()
            (root / 'scripts').mkdir()
            (root / 'output').mkdir()
            fixture = root / 'tests/test_source_snapshots.py'
            original = pre_admission_inputs()['tests/test_source_snapshots.py'].decode()
            fixture.write_text(original)
            (root / 'scripts/ci_qualify_as246.py').write_text('def command():\n    pass\n')
            def git(*args):
                return subprocess.check_output(['git', '-c', 'user.name=CaptureControl', '-c',
                    'user.email=capture-control@example.invalid', *args], cwd=root,
                    stderr=subprocess.PIPE).decode().strip()
            git('init', '-q'); git('add', '.'); git('commit', '-qm', 'retained original fixture')
            pin = git('rev-parse', 'HEAD')
            fixture.write_text(upgrade_source(original))
            git('add', '.'); git('commit', '-qm', 'declared package fixture')
            native_command = inputs.qualifier.command
            def command(argv, *args, **kwargs):
                if argv[:2] == ['git', 'fetch']:
                    return subprocess.CompletedProcess(argv, 0, b'', b'')
                return native_command(argv, *args, **kwargs)
            with patch.object(inputs, 'ROOT', root), patch.object(inputs.qualifier, 'ROOT', root), \
                 patch.object(inputs.qualifier, 'command', side_effect=command):
                self.assertEqual(inputs.require_source_continuity(pin, root / 'output'),
                                 ['tests/test_source_snapshots.py'])
                fixture.write_text(fixture.read_text().replace('self.assertEqual', 'self.assertNotEqual', 1))
                git('add', 'tests/test_source_snapshots.py'); git('commit', '-qm', 'unexpected assertion change')
                with self.assertRaisesRegex(RuntimeError, 'source-fixture drift'):
                    inputs.require_source_continuity(pin, root / 'output')

    def test_missing_runtime_dependency_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.CompletedProcess([], 0, b'libc.so => not found\n', b'')
            with patch.object(inputs.qualifier, 'command', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'native tool dependency closure'):
                    inputs.native_closure({'fixed_backend': {'assembler': '/retained/as'},
                                           'selected_clang': {'path': '/retained/clang'}},
                                          root, root, 'missing')

    def test_real_git_continuity_accepts_only_the_declared_make_enrollment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            (root / 'src').mkdir()
            (root / 'output').mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-c', 'user.name=CaptureControl', '-c',
                    'user.email=capture-control@example.invalid', *args], cwd=root,
                    stderr=subprocess.PIPE).decode().strip()
            git('init', '-q')
            make = root / 'Makefile.gnu'
            make.write_text('gate:\n\t@python3 -m unittest tests.test_source_snapshots tests.test_response_driver_grammar\n')
            source = root / 'src/module_builder.c'
            source.write_text('original native capture source\n')
            (root / 'scripts/ci_qualify_as246.py').write_text('def command():\n    pass\n')
            git('add', '.'); git('commit', '-qm', 'original'); pin = git('rev-parse', 'HEAD')
            make.write_text(make.read_text().replace('tests.test_response_driver_grammar\n',
                'tests.test_response_driver_grammar tests.test_gnu_tool_queries\n'))
            git('add', '.'); git('commit', '-qm', 'declared enrollment')
            native_command = inputs.qualifier.command
            def command(argv, *args, **kwargs):
                if argv[:2] == ['git', 'fetch']:
                    return subprocess.CompletedProcess(argv, 0, b'', b'')
                return native_command(argv, *args, **kwargs)
            with patch.object(inputs, 'ROOT', root), patch.object(inputs.qualifier, 'ROOT', root), \
                 patch.object(inputs.qualifier, 'command', side_effect=command):
                self.assertEqual(inputs.require_source_continuity(pin, root / 'output'), ['Makefile.gnu'])
                make.write_text(make.read_text() + 'CFLAGS += -O0\n')
                git('add', 'Makefile.gnu'); git('commit', '-qm', 'unexpected build flags')
                with self.assertRaisesRegex(RuntimeError, 'beyond the reviewed handoff'):
                    inputs.require_source_continuity(pin, root / 'output')
                source.write_text('changed native capture source\n')
                git('add', 'src'); git('commit', '-qm', 'source drift')
                with self.assertRaisesRegex(RuntimeError, 'unrelated source drift'):
                    inputs.require_source_continuity(pin, root / 'output')


if __name__ == '__main__':
    unittest.main()
