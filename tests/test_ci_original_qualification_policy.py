"""I reject the observer override that changed bootstrap shadows from30 to10."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ci_original_qualification_policy import require_original_timeout_defaults


class OriginalQualificationPolicy(unittest.TestCase):
    def test_unmodified_environment_uses_product_defaults(self):
        require_original_timeout_defaults({})
        require_original_timeout_defaults({'CC': '/usr/bin/cc', 'TMPDIR': '/private/run'})

    def test_ordinary_ten_second_override_is_not_bootstrap_thirty(self):
        with self.assertRaisesRegex(RuntimeError, 'NANO_SHADOW_TIMEOUT_SECONDS'):
            require_original_timeout_defaults({'NANO_SHADOW_TIMEOUT_SECONDS': '10'})

    def test_every_explicit_timeout_override_is_refused(self):
        for key in ['NANO_SHADOW_TIMEOUT_SECONDS', 'BOOTSTRAP_NANOISA_TIMEOUT',
                    'BOOTSTRAP_NANOISA_SHADOW_TIMEOUT', 'CMD_TIMEOUT']:
            for value in ['', '10', '30', '1800', '3600']:
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(RuntimeError, key):
                        require_original_timeout_defaults({key: value})

    def test_workflow_exports_no_timeout_override(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/shape-flow-hosted-arm.yml').read_text())

        def visit(node):
            if isinstance(node, dict):
                if isinstance(node.get('env'), dict):
                    require_original_timeout_defaults(node['env'])
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(workflow)

    def test_both_commands_refuse_override_before_namespace_or_build(self):
        with tempfile.TemporaryDirectory(prefix='original-policy-control-') as temporary:
            base = Path(temporary)
            for name in ['ci_original_native_qualification.py', 'ci_original_consumer20.py']:
                with self.subTest(command=name):
                    output = base / name
                    args = ['--product', str(base / 'no-product'), '--observations', str(output)]
                    if 'consumer20' in name:
                        args += ['--consumer', str(base / 'no-consumer'),
                                 '--native-observations', str(base / 'no-native')]
                    env = {**os.environ, 'NANO_SHADOW_TIMEOUT_SECONDS': '10'}
                    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / name), *args],
                                            env=env, capture_output=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'I require original timeout defaults', result.stderr)
                    self.assertNotIn(b'FileNotFoundError', result.stderr)
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
