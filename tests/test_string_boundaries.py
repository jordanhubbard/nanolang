"""I distinguish literal braces from f-strings without truncating either."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StringBoundaries(unittest.TestCase):
    def test_literal_and_many_part_interpolation(self):
        literal = "a" * 5000 + "{missing}"
        interpolated = "z" * 5000 + "-{value}" * 80 + "-end"
        expected = "z" * 5000 + "-7" * 80 + "-end"
        source_text = (
            'fn main() -> int {\n let value: int = 7\n'
            f' let literal: string = {json.dumps(literal)}\n'
            f' let actual: string = f"{interpolated}"\n'
            f' let expected: string = {json.dumps(expected)}\n'
            ' assert (== (str_length literal) 5009)\n'
            ' assert (== actual expected)\n return 0\n}\n'
            'shadow main { assert (== (main) 0) }\n')
        for compiler, vm in (("nanoc_c", False), ("nano_virt", True)):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-string-boundaries-") as tmp:
                source = Path(tmp) / "strings.nano"
                source.write_text(source_text)
                output = Path(tmp) / ("strings.nvm" if vm else "strings")
                command = [str(ROOT / "bin" / compiler), str(source), "-o", str(output)]
                if vm: command.append("--emit-nvm")
                built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
                self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace")[:2000])
                command = [str(ROOT / "bin/nano_vm"), str(output)] if vm else [str(output)]
                run = subprocess.run(command, capture_output=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace")[:2000])


if __name__ == "__main__":
    unittest.main()
