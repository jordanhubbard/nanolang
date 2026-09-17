"""I retain literal variant identity through all three match execution paths."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InlineUnionMatches(unittest.TestCase):
    def run_case(self, body, expected):
        source = '''union Choice { Some { value: int }, None {} }
fn probe() -> int { BODY }
shadow probe { assert (== (probe) EXPECTED) }
fn main() -> int { assert (== (probe) EXPECTED) return 0 }
'''.replace("BODY", body).replace("EXPECTED", str(expected))
        with tempfile.TemporaryDirectory(prefix="nano-inline-union-") as tmp:
            directory = Path(tmp)
            path = directory / "match.nano"
            binary = directory / "program"
            path.write_text(source)
            commands = [[ROOT / "bin/nano", path],
                        [ROOT / "bin/nano_virt", path, "--run"],
                        [ROOT / "bin/nanoc_c", path, "-o", binary], [binary]]
            for command in commands:
                with self.subTest(command=str(command[0])):
                    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("Error:", result.stderr)
                    self.assertNotIn("Error at", result.stderr)

    def test_expression_payload(self):
        self.run_case('return (match Choice.Some { value: 7 } { '
                      'Some(item) => item.value None(empty) => 0 })', 7)

    def test_statement_side_effect(self):
        self.run_case('let mut result: int = 0 '
                      'match Choice.Some { value: 9 } { '
                      'Some(item) => { set result item.value } '
                      'None(empty) => { set result 1 } } return result', 9)

    def test_lexical_return(self):
        self.run_case('let ignored: int = (match Choice.Some { value: 11 } { '
                      'Some(item) => { return item.value } None(empty) => 0 }) '
                      'return ignored', 11)

    def test_empty_variant(self):
        self.run_case('return (match Choice.None {} { '
                      'Some(item) => item.value None(empty) => 13 })', 13)
