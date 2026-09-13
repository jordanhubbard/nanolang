"""I reject malformed declarations without signals or published artifacts."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ParserErrorRecovery(unittest.TestCase):
    def test_malformed_prefix_arguments(self):
        compiler = os.environ.get("NANOLANG_COMPILER", str(ROOT / "bin/nanoc_c"))
        for expression in ("(+ 1 int)", "(f 1 int)", "(+ (+ 1 2) int)", "(not int)", "(+ 1 2"):
            with self.subTest(expression=expression), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                source = path / "invalid.nano"
                source.write_text("fn main() -> int { return " + expression + " }\n")
                result = subprocess.run([compiler, str(source), "-o", str(path / "program")],
                                        cwd=ROOT, capture_output=True, text=True, timeout=5)
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertFalse((path / "program").exists())

    def test_reserved_local_names(self):
        compiler = os.environ.get("NANOLANG_COMPILER", str(ROOT / "bin/nanoc_c"))
        for name in ("byte", "int", "return", "if"):
            for expression in ("0", "(char_at line 0)"):
                with self.subTest(name=name, expression=expression), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    source = path / "invalid.nano"
                    source.write_text('fn probe(line: string) -> int {\n'
                                      f'let {name}: int = {expression}\n'
                                      f'if (== {name} 34) {{ return 1 }}\n'
                                      'return 0\n}\nfn main() -> int { return 0 }\n')
                    result = subprocess.run([compiler, str(source), "-o", str(path / "program")],
                                            cwd=ROOT, capture_output=True, text=True, timeout=15)
                    self.assertGreater(result.returncode, 0, result.stderr)
                    self.assertFalse((path / "program").exists())
                    self.assertIn("Error", result.stderr)


if __name__ == "__main__":
    unittest.main()
