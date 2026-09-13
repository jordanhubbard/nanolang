"""I check final-expression values, effects and function-scoped returns."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/selfhost/test_match_expression_blocks.nano"


class MatchBlockSemantics(unittest.TestCase):
    def test_reject_wrong_return_and_arm_types(self):
        preamble = FIXTURE.read_text().split("\nfn ", 1)[0]
        bodies = [
            'Some(item) => { if (> item.value 0) { return "bad" } 7 } None(empty) => { 0 }',
            'Some(item) => { 7 } None(empty) => { "bad" }',
            'Some(item) => { 7 } None(empty) => { let unused: int = 0 }',
        ]
        for compiler in [ROOT / "bin/nanoc_c", Path(os.environ.get(
                "NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2"))]:
            for body in bodies:
                with self.subTest(compiler=compiler.name, body=body), tempfile.TemporaryDirectory(
                        prefix="nanolang-match-reject-") as directory:
                    path = Path(directory)
                    source = path / "invalid.nano"
                    source.write_text(preamble + '\nfn broken(choice: Choice) -> int {\n'
                                      'let value = (match choice { ' + body + ' })\nreturn value\n}\n'
                                      'shadow broken { assert true }\nfn main() -> int { return 0 }\n')
                    result = subprocess.run([str(compiler), str(source), "-o", str(path / "invalid")],
                                            cwd=ROOT, env=dict(os.environ, TMPDIR=directory),
                                            capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("error: incompatible", result.stderr,
                                     "I must reject this before invoking the C compiler")

    def test_backend_cases(self):
        fixture = FIXTURE.read_text()
        preamble = fixture.split("\nfn ", 1)[0]
        cases = {"match_local": [(7, 114), (None, 100)],
                 "match_effect": [(7, 107), (None, 200)],
                 "match_branch": [(7, 1), (-7, 102), (None, 103)],
                 "match_string": [(7, 7), (-7, 8), (None, 4)],
                 "match_return": [(7, 7), (None, 103)]}
        compilers = [ROOT / "bin/nanoc_c",
                     Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2"))]
        for compiler in compilers:
            for name, examples in cases.items():
                with self.subTest(compiler=compiler.name, case=name), tempfile.TemporaryDirectory(
                        prefix="nanolang-match-block-") as directory:
                    path = Path(directory)
                    body = fixture.split("\nfn " + name + "(", 1)[1].split("\nshadow " + name, 1)[0]
                    assertions = []
                    for value, expected in examples:
                        choice = "Choice.None { }" if value is None else f"Choice.Some {{ value: {value} }}"
                        assertions.append(f"assert (== ({name} {choice}) {expected})")
                    checks = "\n".join(assertions)
                    source = path / "case.nano"
                    source.write_text(preamble + "\nfn " + name + "(" + body +
                                      "\nshadow " + name + " {\n" + checks + "\n}\n" +
                                      "fn main() -> int {\n" + checks + "\nreturn 0 }\n" +
                                      "shadow main { assert (== (main) 0) }\n")
                    binary = path / "case"
                    compiled = subprocess.run([str(compiler), str(source), "-o", str(binary)],
                                              cwd=ROOT, env=dict(os.environ, TMPDIR=directory),
                                              capture_output=True, text=True, timeout=120)
                    self.assertEqual(compiled.returncode, 0, (compiled.stdout + compiled.stderr)[-4000:])
                    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)


if __name__ == "__main__":
    unittest.main()
