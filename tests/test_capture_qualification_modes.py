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


if __name__ == '__main__':
    unittest.main()
