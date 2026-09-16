"""I check scalar/nested array boundaries and execute their native values."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCALARS = {"int": "7", "float": "1.5", "bool": "true", "string": '"value"'}


class ArrayCompatibility(unittest.TestCase):
    def test_pr294_incremental_cube(self):
        with tempfile.TemporaryDirectory(prefix="nano-incremental-cube-") as d:
            output = Path(d) / "program"
            source = ROOT / "tests/selfhost/test_nested_array_indexing.nano"
            run = subprocess.run([str(ROOT / "bin/nanoc_stage2"), str(source),
                                  "-o", str(output)], cwd=ROOT,
                                 capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            ran = subprocess.run([str(output)], capture_output=True, timeout=10)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_nested_comparison_grouping(self):
        with tempfile.TemporaryDirectory(prefix="nano-comparisons-") as d:
            source = Path(d) / "test.nano"
            output = Path(d) / "program"
            source.write_text('''fn agrees(i: int, j: int) -> bool {
    return (== (== i j) (== j i))
}
shadow agrees { assert (agrees 2 2) assert (agrees 1 2) }
fn main() -> int {
    let mut i: int = 0
    while (< i 4) {
        let mut j: int = 0
        while (< j 4) {
            assert (agrees i j)
            assert (== (< i j) (> j i))
            assert (== (> (+ i 1) j) (< j (+ i 1)))
            set j (+ j 1)
        }
        set i (+ i 1)
    }
    return 0
}
shadow main { assert (== (main) 0) }
''')
            run = subprocess.run([str(ROOT / "bin/nanoc_stage2"), str(source),
                                  "-o", str(output)], cwd=ROOT,
                                 capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            ran = subprocess.run([str(output)], capture_output=True, timeout=10)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_scalar_mismatches(self):
        self.check_mismatches(SCALARS)

    def test_nested_mismatches(self):
        self.check_mismatches({"array<int>": "[7]", "array<float>": "[1.5]",
                               "array<array<int>>": "[[7]]"})

    def test_nested_values(self):
        with tempfile.TemporaryDirectory(prefix="nano-nested-values-") as d:
            source = Path(d) / "test.nano"
            output = Path(d) / "program"
            source.write_text('''fn rows() -> array<array<int>> {
    return [[7, 8], [9], []]
}
shadow rows { assert (== (array_length (rows)) 3) }
fn main() -> int {
    let values = (rows)
    assert (== (at (at values 0) 1) 8)
    assert (== (at (at values 1) 0) 9)
    assert (== (array_length (at values 2)) 0)
    let deep: array<array<array<int>>> = [[[11], [12, 13]], [[14]]]
    assert (== (at (at (at deep 0) 1) 1) 13)
    assert (== (at (at (at deep 1) 0) 0) 14)
    let empty: array<array<int>> = []
    let appended = (array_push empty [17, 18])
    assert (== (at (at appended 0) 1) 18)
    let fractional: array<array<float>> = [[], [1.5]]
    let filled = (array_push (at fractional 0) 2.5)
    assert (== (at filled 0) 2.5)
    assert (== (at (at fractional 1) 0) 1.5)
    return 0
}
shadow main { assert (== (main) 0) }
''')
            run = subprocess.run([str(ROOT / "bin/nanoc_stage2"), str(source),
                                  "-o", str(output)], cwd=ROOT,
                                 capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            ran = subprocess.run([str(output)], capture_output=True, timeout=10)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def check_mismatches(self, types):
        for source_type, value in types.items():
            for target_type in types:
                if source_type == target_type:
                    continue
                for route in ("alias", "argument", "return", "set"):
                    with self.subTest(source=source_type, target=target_type, route=route), \
                         tempfile.TemporaryDirectory(prefix="nano-array-types-") as d:
                        directory = Path(d)
                        source = directory / "test.nano"
                        output = directory / "prior.c"
                        report = directory / "diagnostics.json"
                        prefix = f'''fn produce() -> array<{source_type}> {{ return [{value}] }}
shadow produce {{ assert (== (array_length (produce)) 1) }}
fn accept(xs: array<{target_type}>) -> int {{ return (array_length xs) }}
shadow accept {{ assert (== (accept []) 0) }}
'''
                        body = {
                            "alias": f"let wrong: array<{target_type}> = source",
                            "argument": "let wrong: int = (accept source)",
                            "return": "let wrong = (bad)",
                            "set": f"let mut wrong: array<{target_type}> = [] set wrong source",
                        }[route]
                        if route == "return":
                            prefix += f"fn bad() -> array<{target_type}> {{ return (produce) }}\nshadow bad {{ assert true }}\n"
                        source.write_text(prefix + f"fn main() -> int {{ let source: array<{source_type}> = (produce) {body} return 0 }}\nshadow main {{ assert true }}\n")
                        output.write_bytes(b"prior artifact")
                        run = subprocess.run([str(ROOT / "bin/nanoc_stage2"), str(source),
                                              "--target", "c", "-o", str(output),
                                              "--llm-diags-json", str(report)], cwd=ROOT,
                                             capture_output=True, timeout=60)
                        self.assertGreater(run.returncode, 0, run.stdout + run.stderr)
                        code = "E0010" if route == "argument" else "E0001"
                        diagnostics = json.loads(report.read_text())["diagnostics"]
                        boundary = {"alias": "Variable wrong", "argument": "Argument 1",
                                    "return": "Return value of bad", "set": "Assignment to wrong"}[route]
                        self.assertTrue(any(item["code"] == code and boundary in item["message"]
                                            for item in diagnostics), diagnostics)
                        self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_scalar_identity_and_empty_arrays(self):
        for kind, value in SCALARS.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="nano-array-valid-") as d:
                source = Path(d) / "test.nano"
                output = Path(d) / "program"
                source.write_text(f'''fn identity(xs: array<{kind}>) -> array<{kind}> {{ return xs }}
shadow identity {{ assert (== (array_length (identity [])) 0) }}
fn main() -> int {{
    let empty: array<{kind}> = []
    let filled: array<{kind}> = (array_push empty {value})
    let values: array<{kind}> = (identity filled)
    assert (== (at values 0) {value})
    return 0
}}
shadow main {{ assert (== (main) 0) }}
''')
                run = subprocess.run([str(ROOT / "bin/nanoc_stage2"), str(source),
                                      "-o", str(output)], cwd=ROOT,
                                     capture_output=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                ran = subprocess.run([str(output)], capture_output=True, timeout=10)
                self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)


if __name__ == "__main__":
    unittest.main()
