"""I exercise the real installation recipe with synthetic bootstrap artifacts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from scripts.bootstrap_nanoisa import Bootstrap, digest, source_inputs

ROOT = Path(__file__).resolve().parents[1]
COMPILER = '''#!/usr/bin/env python3
import pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
out.write_text("#!/bin/sh\\nexit 0\\n")
out.chmod(0o755)
'''


class BootstrapMessages(unittest.TestCase):
    def run_branch(self, identical, deterministic=False, dependency_verifier='/usr/bin/true'):
        with tempfile.TemporaryDirectory(prefix='nanolang-bootstrap-messages-') as directory:
            root = Path(directory)
            for name in ('Makefile.gnu', 'scripts/bootstrap_nanoisa.py',
                         'tests/bootstrap_native_guard.py'):
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, destination)
            (root / 'bin').mkdir()
            (root / 'work').mkdir()
            (root / '.bootstrap2.built').touch()
            (root / '.bootstrap3.built').touch()
            for name in ('nano_virt', 'nano_vm', 'nanoisa', 'nvm2c', 'nano_aot_runtime.o', 'nano_as_capture.so'):
                path = root / 'bin' / name
                path.write_text('#!/bin/sh\nexit 0\n')
                path.chmod(0o755)
            # Different native bytes must not affect the raw module contract.
            for name, suffix in [('nanoc_stage1', ''), ('nanoc_stage2', '# Different native bytes\n')]:
                path = root / 'bin' / name
                path.write_text(COMPILER + suffix)
                path.chmod(0o755)
            for name in ('seed', 'stage1', 'stage2'):
                (root / 'bin' / f'nanoc_{name}.nvm').write_bytes(
                    b'raw module' if name != 'stage2' or identical else b'different raw module')
            artifacts = ['nanoc_seed.nvm', 'nanoc_stage1.nvm', 'nanoc_stage2.nvm',
                         'nanoc_stage1', 'nanoc_stage2']
            receipt = {'version': 1, 'root': str(root.resolve()), 'work': str((root / 'work').resolve()),
                       'phase': 'stage2', 'sources': source_inputs(root), 'tools': Bootstrap(root, 30).tool_inputs(),
                       'hosts': {}, 'artifacts': {name: digest(root / 'bin' / name) for name in artifacts},
                       'steps': []}
            (root / 'bin/nanoc_bootstrap.json').write_text(json.dumps(receipt))
            result = subprocess.run(
                [os.environ.get('MAKE_BIN', 'make'), '-s', '-f', 'Makefile.gnu',
                 '-o', '.bootstrap2.built', '-W', '.bootstrap2.built', '.bootstrap3.built', 'UNAME_S=Linux',
                 'VERIFY_SCRIPT=' + dependency_verifier, f'BOOTSTRAP_TMPDIR={root}',
                 f'BOOTSTRAP_DETERMINISTIC={int(deterministic)}'],
                cwd=root, capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout + result.stderr, (root / '.bootstrap3.built').exists()

    def test_equal_modules_allow_different_native_bytes_without_correctness_claim(self):
        status, output, stamped = self.run_branch(True)
        self.assertEqual(status, 0, output)
        self.assertTrue(stamped)
        self.assertIn('raw Stage 1 and Stage 2 modules: they are byte-identical', output)
        self.assertIn('not established reproducibility across clean environments', output)
        self.assertNotIn('This proves reproducible builds', output)

    def test_different_raw_modules_always_refuse_installation(self):
        for deterministic in (False, True):
            with self.subTest(deterministic=deterministic):
                status, output, stamped = self.run_branch(False, deterministic)
                self.assertNotEqual(status, 0, output)
                self.assertFalse(stamped)
                self.assertIn('I require byte-identical raw Stage 1 and Stage 2 modules', output)

    def test_failed_installed_dependency_check_does_not_stamp_success(self):
        status, output, stamped = self.run_branch(True, dependency_verifier='/usr/bin/false')
        self.assertNotEqual(status, 0, output)
        self.assertFalse(stamped)
        self.assertIn('I installed bin/nanoc', output)


if __name__ == '__main__':
    unittest.main()
