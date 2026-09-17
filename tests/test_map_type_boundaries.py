"""I reject map tag changes before either compiler publishes output."""
from itertools import product
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MapTypeBoundaries(unittest.TestCase):
    def test_mismatches_preserve_prior_artifact(self):
        pairs = list(product(("int", "string"), repeat=2))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "case.nano"
            output = Path(tmp) / "prior"
            for actual, wanted in product(pairs, repeat=2):
                if actual == wanted:
                    continue
                a, w = ("HashMap<" + ",".join(pair) + ">" for pair in (actual, wanted))
                cases = {
                    "return": f"fn bad(m: {a}) -> {w} {{ return m }}",
                    "call_return": f"fn bad() -> {w} {{ return (fresh) }}",
                    "binding": f"fn bad(m: {a}) -> void {{ let target: {w} = m }}",
                    "argument": f"fn take(m: {w}) -> void {{ }} fn bad(m: {a}) -> void {{ (take m) }}",
                    "assignment": f"fn bad(m: {a}) -> void {{ let mut target: {w} = (map_new) set target m }}",
                    "global": f"let target: {w} = (fresh)",
                    "record": f"struct Holder {{ values: {w} }} fn bad(m: {a}) -> void {{ let h: Holder = Holder {{ values: m }} }}",
                    "conditional": f"fn bad(m: {a}) -> {w} {{ return (cond (true m) (else (map_new))) }}",
                }
                for boundary, body in cases.items():
                    source.write_text(
                        f"fn fresh() -> {a} {{ let m: {a} = (map_new) return m }}\n"
                        + body + "\nfn main() -> int { return 0 } shadow main { assert true }\n"
                    )
                    for compiler in ("nanoc_c", "nano_virt"):
                        with self.subTest(actual=actual, wanted=wanted, boundary=boundary, compiler=compiler):
                            output.write_text("prior artifact")
                            command = [ROOT / "bin" / compiler, source, "-o", output]
                            if compiler == "nano_virt":
                                command.append("--emit-nvm")
                            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
                            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertIn("Match both declared map key and value types.", result.stderr)
                            self.assertEqual(output.read_text(), "prior artifact")


if __name__ == "__main__":
    unittest.main()
