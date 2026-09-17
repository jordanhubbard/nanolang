"""I evaluate callees before argument side effects on interpreter and VM paths."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

BINDING = 'union Choice { Some { value: int }, None { } }\nfn first(x: int) -> int { return (+ x 100) }\nshadow first { assert (== (first 1) 101) }\nfn second(x: int) -> int { return (+ x 200) }\nshadow second { assert (== (second 1) 201) }\nfn main() -> int {\n let mut target: fn(int) -> int = first\n let choice: Choice = Choice.Some { value: 1 }\n let result: int = (target (match choice { Some(item) => { set target second 1 } None(empty) => { 0 } }))\n assert (== result 101)\n assert (== (target 1) 201)\n return 0\n}\nshadow main { assert true }\n'
COMPUTED = 'let mut trace: int = 0\nfn add(x: int) -> int { return (+ x 100) }\nshadow add { assert (== (add 1) 101) }\nfn choose() -> fn(int) -> int {\n    set trace (+ (* trace 10) 1)\n    return add\n}\nshadow choose { assert (== ((choose) 1) 101) }\nfn argument() -> int {\n    set trace (+ (* trace 10) 2)\n    return 1\n}\nshadow argument { assert (== (argument) 1) }\nfn main() -> int {\n    set trace 0\n    assert (== ((choose) (argument)) 101)\n    assert (== trace 12)\n    return 0\n}\nshadow main { assert true }\n'

class CalleeSnapshots(unittest.TestCase):
    def check_backends(self, source):
        with tempfile.TemporaryDirectory(prefix="nano-callee-snapshot-") as tmp:
            path = Path(tmp) / "main.nano"
            path.write_text(source)
            for binary, flags in [("nano", []), ("nano_virt", ["--run"])]:
                with self.subTest(backend=binary):
                    result = subprocess.run([ROOT / "bin" / binary, path, *flags],
                                            capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("Error:", result.stderr)

    def test_mutated_function_binding(self):
        self.check_backends(BINDING)

    def test_computed_callee_precedes_arguments(self):
        self.check_backends(COMPUTED)
