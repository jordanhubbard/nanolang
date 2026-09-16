"""I require map type errors before native C compilation."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = '''fn convert(n: int) -> float { return 1.5 }
shadow convert { assert (== (convert 1) 1.5) }
fn choose() -> fn(int) -> float { return convert }
shadow choose { assert (== ((choose) 1) 1.5) }
fn pair(a: int, b: int) -> float { return 1.5 }
shadow pair { assert (== (pair 1 2) 1.5) }
fn discard(n: int) -> void { }
shadow discard { (discard 1) }
'''


class MapTypes(unittest.TestCase):
    def test_inferred_result_executes(self):
        with tempfile.TemporaryDirectory(prefix="nano-map-infer-") as d:
            source = Path(d) / "test.nano"
            output = Path(d) / "program"
            source.write_text(PREFIX + '''fn main() -> int {
    let values = (map [1, 2] (choose))
    assert (== (array_length values) 2)
    assert (== (at values 0) 1.5)
    return 0
}
shadow main { assert (== (main) 0) }
''')
            run = subprocess.run([str(ROOT / "bin/nanoc_stage2"), str(source),
                                  "-o", str(output)], cwd=ROOT,
                                 capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = subprocess.run([str(output)], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejections_before_c_emission(self):
        cases = {
            "arity": ("let value = (map [1])", "a source array and one transform"),
            "source": ("let value = (map 1 convert)", "an array source in map"),
            "callback": ("let value = (map [1] 7)", "a function transform in map"),
            "callback_arity": ("let value = (map [1] pair)", "a unary transform in map"),
            "callback_void": ("let value = (map [1] discard)", "a known value-producing transform in map"),
            "input": ("let value = (map [1.5] convert)", "map input elements to match"),
            "returned_input": ("let value = (map [1.5] (choose))", "map input elements to match"),
            "annotation": ("let value: array<int> = (map [1] convert)", "declared map array type to match"),
        }
        for name, (statement, diagnostic) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="nano-map-types-") as d:
                source = Path(d) / "test.nano"
                output = Path(d) / "test.c"
                source.write_text(PREFIX + "fn main() -> int { " + statement +
                                  " return 0 }\nshadow main { assert true }\n")
                output.write_bytes(b"prior artifact")
                run = subprocess.run([str(ROOT / "bin/nanoc_stage2"), str(source),
                                      "--target", "c", "-o", str(output)], cwd=ROOT,
                                     capture_output=True, timeout=60)
                self.assertGreater(run.returncode, 0, run.stdout + run.stderr)
                self.assertIn(diagnostic.encode(), run.stdout + run.stderr)
                self.assertEqual(output.read_bytes(), b"prior artifact")


if __name__ == "__main__":
    unittest.main()
