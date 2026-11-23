"""I compare actual retained Python and shell package-fixture behavior."""
import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest

from tests.assembler_search_fixture import ADDED_IMPORT, LEGACY, REPLACEMENT, pkg_config_script, upgrade_source
from tests.retained_capture_evidence import pre_admission_inputs
from tests.native_process import run_captured


@unittest.skipUnless(os.name == 'posix', 'I exercise the POSIX source fixture.')
class AssemblerSearchFixture(unittest.TestCase):
    def test_actual_legacy_and_shell_protocol_outputs_match(self):
        source = pre_admission_inputs()['tests/test_source_snapshots.py'].decode()
        tree = ast.parse(source)
        expression = next(node.args[0] for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text'
            and isinstance(node.args[0], ast.JoinedStr)
            and 'mapping' in ast.unparse(node.args[0]) and '--cflags' in ast.unparse(node.args[0]))
        mapping = {'assembler-search-fixture-0': '-I /tmp/early includes',
                   'assembler-search-fixture-1': "'-I/tmp/with quotes'",
                   'assembler-search-fixture-2': '-Wa,path\\with\\slashes',
                   'assembler-search-fixture-3': 'λ $HOME $(printf forbidden) `false`\n-O2',
                   'assembler-search-fixture-4': '',
                   'assembler-search-fixture-5': '--alternate'}
        legacy = eval(compile(ast.Expression(expression), '<retained fixture>', 'eval'), {'sys': sys, 'mapping': mapping})
        with tempfile.TemporaryDirectory(prefix='nano-package-protocol-') as directory:
            root = Path(directory)
            old, new = root / 'python-pkg-config', root / 'shell-pkg-config'
            old.write_text(legacy); new.write_text(pkg_config_script(mapping))
            old.chmod(0o700); new.chmod(0o700)
            queries = [(mode, '--', name) for name in mapping for mode in ('--exists', '--cflags', '--libs')]
            queries += [(), ('--libs', 'unknown'), ('--cflags', '--', 'unknown'),
                        ('--libs', '--cflags', '--', 'assembler-search-fixture-3')]
            for query in queries:
                with self.subTest(query=query):
                    before = run_captured([str(old), *query], timeout=5)
                    after = run_captured([str(new), *query], timeout=5)
                    self.assertEqual((after.returncode, after.stdout), (before.returncode, before.stdout))
                    if before.returncode == 0:
                        self.assertEqual(after.stderr, before.stderr)

    def test_only_declared_fixture_bytes_change_and_all_assertions_remain(self):
        source = pre_admission_inputs()['tests/test_source_snapshots.py'].decode()
        changed = upgrade_source(source)
        self.assertEqual(changed.replace(ADDED_IMPORT, '').replace(REPLACEMENT, LEGACY), source)
        def assertions(text):
            return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(text))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr.startswith(('assert', 'skipTest'))]
        self.assertEqual(assertions(source), assertions(changed))
        methods = lambda text: [node.name for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]
        self.assertEqual(methods(source), methods(changed))
        with self.assertRaises(RuntimeError):
            upgrade_source(source.replace(LEGACY, 'unexpected fixture'))


if __name__ == '__main__':
    unittest.main()
