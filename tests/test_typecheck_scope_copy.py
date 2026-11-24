"""I preserve my scope-copy metadata and original complete checker diagnostics."""
from pathlib import Path
import json
import os
import shutil
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked

ROOT = Path(__file__).resolve().parents[1]


class TypecheckScopeCopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report_dir = os.environ.get('NANO_TYPECHECK_REPORT_DIR')
        cls.directory = Path(tempfile.mkdtemp(prefix='nano-scope-copy-', dir=report_dir))
        if not report_dir:
            cls.addClassCleanup(shutil.rmtree, cls.directory)
        cls.corpus = json.loads((ROOT / 'tests/typecheck_scope_copy_cases.json').read_text())
        if len(cls.corpus['cases']) != 24:
            raise AssertionError('I require my complete 24-input checker corpus.')
        cls.routes = build_routes(ROOT / 'tests/typecheck_scope_copy.nano', cls.directory)

    def test_copy_metadata_aliases_and_growth(self):
        for route_name, command in self.routes:
            with self.subTest(route=route_name):
                self.assertEqual(checked(command), 'SCOPE-COPY-OK\n')

    def test_original_complete_diagnostic_records(self):
        names = [case['name'] for case in self.corpus['cases']]
        self.assertEqual(len(set(names)), 24)
        for index, case in enumerate(self.corpus['cases']):
            source = self.directory / f'case-{index}.input'
            source.write_bytes(bytes.fromhex(case['source_hex']))
            for route_name, command in self.routes:
                with self.subTest(case=case['name'], route=route_name):
                    self.assertEqual(checked([*command, source]), case['expected'])


if __name__ == '__main__':
    unittest.main()
