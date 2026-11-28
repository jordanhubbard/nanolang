"""I retain complete raw lowering reports through my original producer routes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tests import nanolang_translator_routes as routes

ROOT = Path(__file__).resolve().parents[1]


class NanoisaLookupProjection(unittest.TestCase):
    def test_controls_and_complete_lowering_reports(self):
        cases = json.loads((ROOT/'tests/typecheck_scope_copy_cases.json').read_text())['cases']
        expected = json.loads((ROOT/'tests/nanoisa_lookup_projection_cases.json').read_text())
        self.assertEqual(len(cases), 24)
        self.assertEqual(len(expected), len(cases))
        with tempfile.TemporaryDirectory(prefix='nano-lookup-projection-') as temporary:
            directory = Path(temporary)
            products = routes.build_routes(ROOT/'tests/nanoisa_lookup_projection.nano', directory)
            self.assertEqual(len(products), 5 if routes.source_compiler_command() else 3)
            for label, command in products:
                with self.subTest(route=label):
                    self.assertEqual(routes.checked(command), 'LOOKUP-PROJECTION-OK\n')
            for index, (case, wanted) in enumerate(zip(cases, expected)):
                source = bytes.fromhex(case['source_hex'])
                self.assertEqual(hashlib.sha256(source).hexdigest(), wanted['source_sha256'])
                path = directory/f'case-{index}.nano'
                path.write_bytes(source)
                for label, command in products:
                    with self.subTest(case=index, route=label):
                        actual = routes.checked([*command, path])
                        self.assertEqual(hashlib.sha256(actual.encode()).hexdigest(),
                                         wanted['report_sha256'], actual)


if __name__ == '__main__':
    unittest.main()
