"""I compare complete lexer records with my retained pre-snapshot behavior."""
from pathlib import Path
import json
import os
import shutil
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked

ROOT = Path(__file__).resolve().parents[1]


class LexerSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report_dir = os.environ.get('NANO_LEXER_REPORT_DIR')
        cls.directory = Path(tempfile.mkdtemp(prefix='nano-lexer-snapshot-', dir=report_dir))
        if not report_dir:
            cls.addClassCleanup(shutil.rmtree, cls.directory)
        cls.corpus = json.loads((ROOT / 'tests/lexer_snapshot_cases.json').read_text())
        if len(cls.corpus['cases']) != 24:
            raise AssertionError('I require my complete 24-input corpus.')
        cls.routes = build_routes(ROOT / 'tests/lexer_snapshot_report.nano', cls.directory)

    def test_complete_token_and_diagnostic_records(self):
        names = [case['name'] for case in self.corpus['cases']]
        self.assertEqual(len(set(names)), 24)
        for index, case in enumerate(self.corpus['cases']):
            path = self.directory / f'case-{index}.input'
            path.write_bytes(bytes.fromhex(case['source_hex']))
            for route_name, command in self.routes:
                with self.subTest(case=case['name'], route=route_name):
                    self.assertEqual(checked([*command, path]), case['expected'])


if __name__ == '__main__':
    unittest.main()
