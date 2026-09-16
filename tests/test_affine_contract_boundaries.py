"""Source-level regressions for affine type-checking boundaries."""
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "bin" / "nanoc_c"


class AffineContractBoundaryTests(unittest.TestCase):
    def compile_source(self, source: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scope.nano"
            path.write_text(textwrap.dedent(source))
            return subprocess.run(
                [str(COMPILER), str(path), "-o", str(Path(tmp) / "scope")],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

    def test_shadowed_ordinary_binding(self):
        result = self.compile_source(
            """
            resource struct FileHandle { fd: int }
            fn consume_handle(file: FileHandle) -> void { }

            fn probe(file: FileHandle, choose: bool) -> void {
                if choose { let file: int = 3 assert (== file 3) }
                if choose {
                    let file: int = 4
                    assert (== file 4)
                } else { let file: int = 5 assert (== file 5) }
                while false { let file: int = 6 assert (== file 6) }
                for file in (range 0 1) { assert (== file 0) }
                fn nested(file: int) -> void { assert (== file file) }
                unsafe { (consume_handle file) }
            }

            fn main() -> int { return 0 }
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
