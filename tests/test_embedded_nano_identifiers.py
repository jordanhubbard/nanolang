"""I catch the reported fixture regressions with my real lexer, before bootstrap."""
import os
from pathlib import Path
import tempfile
import unittest

from scripts.check_embedded_nano_identifiers import literals, scan

ROOT = Path(__file__).resolve().parents[1]
LEXER = Path(os.environ.get('NANO_FIXTURE_LEXER', ROOT/'obj/fixture_identifier_lexer'))


class EmbeddedNanoIdentifiers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fixture-keywords-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'fixture.py'

    def fixture(self, program, expected=None):
        marker = f'# nano-fixture: expected-reserved-name={expected}\n' if expected else ''
        self.path.write_text(marker + f'program = {program!r}\n')
        return scan(LEXER, [self.path], jobs=1)

    def test_strings_comments_and_mutable_names_follow_actual_lexer(self):
        findings, checked, refused = self.fixture('''fn good() -> int {
    let mut value: int = 7
    (println "fn effect() let handle: int = 0")
    # let handle: int = 0
    /* fn effect() */
    return value
}
shadow good { assert (== (good) 7) }
''')
        self.assertEqual((findings, checked, refused), ([], 1, 0))

    def test_real_reported_fixtures_refuse_the_original_regressions(self):
        controls = [
            ('test_nanoisa_shadow_emitter.py', 'let owner: Handle',
             # nano-fixture: expected-reserved-name=handle
             'let handle: Handle', 'handle'),
            ('test_nanoisa_flat_records.py', 'fn unused()',
             # nano-fixture: expected-reserved-name=effect
             'fn effect()', 'effect'),
        ]
        for filename, before, after, name in controls:
            with self.subTest(name=name):
                original = ROOT/'tests'/filename
                clean, _, _ = scan(LEXER, [original], jobs=1)
                self.assertEqual(clean, [])
                text = original.read_text()
                self.assertIn(before, text)
                copy = self.path.with_name(filename)
                copy.write_text(text.replace(before, after, 1))
                findings, _, _ = scan(LEXER, [copy], jobs=1)
                self.assertIn(name, [item['name'] for item in findings])
                self.assertTrue(all(item['literal_line'] > 0 and item['nano_line'] > 0 for item in findings))

    def test_exact_negative_annotation_does_not_hide_other_reserved_names(self):
        name, other = 'not', 'effect'
        program = f'fn good() -> int {{ let {name}: int = 7 return 0 }}'
        findings, _, _ = self.fixture(program, expected=name)
        self.assertEqual(findings, [])
        findings, _, _ = self.fixture(program.replace('return 0', f'let {other}: int = 9 return 0'), expected=name)
        self.assertEqual([item['name'] for item in findings], [other])
        with self.assertRaisesRegex(RuntimeError, 'annotated reserved name'):
            self.fixture(program.replace(name, 'ordinary'), expected=name)

    def test_docstrings_are_not_fixture_programs_and_literals_are_not_executed(self):
        name = 'handle'
        self.path.write_text(f'"""fn {name}() is an intentionally bad example in prose."""\n'
                             'raise RuntimeError("I must not execute Python fixture code")\n'
                             'program = "fn main() -> int { return 0 } shadow main { assert (== (main) 0) }"\n')
        self.assertEqual(len(literals(self.path)), 1)
        findings, checked, refused = scan(LEXER, [self.path], jobs=1)
        self.assertEqual((findings, checked, refused), ([], 1, 0))


if __name__ == '__main__':
    unittest.main()
