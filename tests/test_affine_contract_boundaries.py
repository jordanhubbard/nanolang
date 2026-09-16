"""I test ownership decisions without treating a field read as destruction."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = """resource struct FileHandle { fd: int }
extern fn consume_handle(owned: FileHandle) -> void
"""
OWNERSHIP = r"(?i)(ownership|resource.{0,80}(scope|leak|live|consum)|moved value|after.{0,30}(mov|consum))"


class AffineContractBoundaries(unittest.TestCase):
    def check_case(self, name, declaration, accepted):
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            with self.subTest(case=name, compiler=compiler), tempfile.TemporaryDirectory(
                prefix="nano-affine-contract-"
            ) as directory:
                source = Path(directory) / "case.nano"
                output = Path(directory) / "case.c"
                # These are declaration/typecheck probes, not shadow execution
                # or a claim that the foreign operation has been implemented.
                source.write_text(PREFIX + declaration + "\nfn main() -> int { return 0 }\n")
                output.write_bytes(b"prior artifact")
                result = subprocess.run(
                    [str(ROOT / "bin" / compiler), str(source), "--target", "c", "-o", str(output)],
                    cwd=ROOT, capture_output=True, text=True, timeout=120,
                )
                diagnostic = result.stdout + result.stderr
                if accepted:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    self.assertNotEqual(output.read_bytes(), b"prior artifact")
                else:
                    self.assertGreater(result.returncode, 0, diagnostic)
                    self.assertRegex(diagnostic, OWNERSHIP)
                    self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_return_owned_parameter(self):
        self.check_case("return_parameter", "fn probe(file: FileHandle) -> FileHandle { return file }", True)

    def test_move_then_consume(self):
        self.check_case("move_consume", """fn probe(first: FileHandle) -> void {
    let second: FileHandle = first
    unsafe { (consume_handle second) }
}""", True)

    def test_observe_then_return_owner(self):
        self.check_case("observe_return", """fn probe(file: FileHandle) -> FileHandle {
    let observed: int = file.fd
    return file
}""", True)

    def test_field_observation_does_not_resolve_parameter(self):
        self.check_case("observe_leak", "fn probe(file: FileHandle) -> int { return file.fd }", False)

    def test_unused_parameter_is_unresolved(self):
        self.check_case("parameter_leak", "fn probe(file: FileHandle) -> int { return 0 }", False)

    def test_observation_after_move(self):
        self.check_case("moved_observation", """fn probe(first: FileHandle) -> int {
    let second: FileHandle = first
    unsafe { (consume_handle second) }
    return first.fd
}""", False)

    def test_owner_beyond_old_fixed_capacity(self):
        statements = []
        for index in range(256):
            statements.append(f"let owner_{index}: FileHandle = FileHandle {{ fd: {index} }}")
            statements.append(f"unsafe {{ (consume_handle owner_{index}) }}")
        statements.append("let remaining: FileHandle = FileHandle { fd: 256 }")
        self.check_case("owner_257", "fn probe() -> int {\n" + "\n".join(statements) + "\nreturn 0\n}", False)


if __name__ == "__main__":
    unittest.main()
