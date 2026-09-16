"""I compile and execute physical multiline strings across module boundaries."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve()


class ModuleMultilineStrings(unittest.TestCase):
    def test_multiline_literal_preserves_next_module_owner(self):
        with tempfile.TemporaryDirectory(prefix="nano-multiline-module-") as tmp:
            directory = Path(tmp)
            first = directory / "first.nano"
            second = directory / "second.nano"
            main = directory / "main.nano"
            output = directory / "program"

            first.write_text('pub fn text() -> string { return "one\ntwo\nthree\nfour\nfive" }\n'
                             'shadow text { assert (> (str_length (text)) 0) }\n')
            second.write_text(f'module "{first}" as first\n'
                              'pub fn answer() -> int { return (str_length (first.text)) }\n'
                              'shadow answer { assert (> (answer) 0) }\n')
            main.write_text(f'module "{second}" as second\n'
                            'let banner: string = "a\nb"\n'
                            'fn main() -> int {\n'
                            '    assert (== (second.answer) 23)\n'
                            '    assert (== (str_length banner) 3)\n'
                            '    assert (== (char_at banner 1) 10)\n'
                            '    return 0\n'
                            '}\n'
                            'shadow main { assert (== (main) 0) }\n')

            compiled = subprocess.run(
                [str(COMPILER), str(main), "-o", str(output)],
                cwd=ROOT,
                capture_output=True,
                timeout=90,
                env=dict(os.environ, TMPDIR=tmp),
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            executed = subprocess.run([str(output)], capture_output=True, timeout=10)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main()
