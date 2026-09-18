"""I retain exact conditional field types through canonical lowering."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ConditionalFieldTypes(unittest.TestCase):
    def check(self, field_type, left, right, assertion, reject=False):
        source = f"""struct Holder {{ value: {field_type} }}
fn choose(flag: bool) -> Holder {{
 return Holder {{ value: (cond (flag {left}) (else {right})) }}
}}
shadow choose {{ {assertion} }}
fn main() -> int {{ {assertion} return 0 }}
shadow main {{ assert (== (main) 0) }}
"""
        with tempfile.TemporaryDirectory(prefix="nano-conditional-fields-") as directory:
            path = Path(directory) / "input.nano"
            path.write_text(source)
            for compiler in ("nano_virt", "nanoc_stage1", "nanoc_stage2"):
                with self.subTest(compiler=compiler):
                    output = Path(directory) / (compiler + ".nvm")
                    output.write_bytes(b"previous accepted artifact")
                    result = subprocess.run([ROOT / "bin" / compiler, path, "--emit-nvm", "-o", output], cwd=ROOT, capture_output=True, text=True, timeout=150)
                    diagnostic = result.stdout + result.stderr
                    if reject:
                        self.assertGreater(result.returncode, 0, diagnostic)
                        self.assertEqual(output.read_bytes(), b"previous accepted artifact")
                    else:
                        self.assertEqual(result.returncode, 0, diagnostic)
                        run = subprocess.run([ROOT / "bin/nano_vm", output], cwd=ROOT, capture_output=True, text=True, timeout=15)
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_string_arms_and_nested_choice(self):
        self.check("string", '\"first\"', '(cond (false \"unused\") (else \"last\"))', 'assert (== (choose true).value \"first\") assert (== (choose false).value \"last\")')

    def test_boolean_arms(self):
        self.check("bool", "true", "false", "assert (choose true).value assert (not (choose false).value)")

    def test_integer_arms(self):
        self.check("int", "7", "9", "assert (== (choose true).value 7) assert (== (choose false).value 9)")

    def test_mismatched_arms_preserve_output(self):
        self.check("string", '\"first\"', "7", "assert true", reject=True)

    def test_wrong_declared_field_preserves_output(self):
        self.check("int", '\"first\"', '\"last\"', "assert true", reject=True)

if __name__ == "__main__":
    unittest.main()
