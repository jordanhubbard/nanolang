"""I require each frontend to make the same checked ownership decisions."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AffineFrontendParity(unittest.TestCase):
    def test_existing_ownership_cases(self):
        cases = (("valid_move.nano", None),
                 ("use_after_move.nano", r"(?i)(after.{0,30}(mov|consum)|already consumed|moved value)"),
                 ("unresolved.nano", r"(?i)(resource.{0,80}(scope|leak|live|resolv|consum)|unresolved)"))
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            for filename, diagnostic in cases:
                with self.subTest(compiler=compiler, case=filename), tempfile.TemporaryDirectory(prefix="nano-affine-parity-") as tmp:
                    output = Path(tmp) / "program"
                    output.write_bytes(b"prior artifact")
                    result = subprocess.run(
                        [str(ROOT / "bin" / compiler), str(ROOT / "tests/affine_selfhost" / filename), "-o", str(output)],
                        cwd=ROOT, capture_output=True, text=True, timeout=120)
                    messages = result.stdout + result.stderr
                    if diagnostic is None:
                        self.assertEqual(result.returncode, 0, messages)
                        execution = subprocess.run([str(output)], capture_output=True, timeout=10)
                        self.assertEqual(execution.returncode, 42, execution.stderr)
                    else:
                        self.assertGreater(result.returncode, 0, messages)
                        self.assertRegex(messages, diagnostic)
                        self.assertEqual(output.read_bytes(), b"prior artifact")


if __name__ == "__main__":
    unittest.main()
