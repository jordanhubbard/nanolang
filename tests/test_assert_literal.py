"""I print assertion source as text, not as a native format string."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AssertionLiteral(unittest.TestCase):
    def test_percent_directives_are_literal(self):
        source = '''
fn check(text: string) -> void {
    assert (== text "%s %d %n %%")
}
shadow check { (check "%s %d %n %%") }
fn main() -> int { (check "different") return 0 }
'''
        with tempfile.TemporaryDirectory(prefix="nano-assert-literal-") as tmp:
            path = Path(tmp) / "assert.nano"
            path.write_text(source)
            output = Path(tmp) / "program"
            compiled = subprocess.run([str(ROOT / "bin/nanoc_c"), str(path),
                                       "-o", str(output)], cwd=ROOT,
                                      capture_output=True, text=True, timeout=60)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            run = subprocess.run([str(output)], capture_output=True,
                                 text=True, timeout=10)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            self.assertIn('%s %d %n %%', run.stderr)
            self.assertIn('Contract violation', run.stderr)


if __name__ == "__main__":
    unittest.main()
