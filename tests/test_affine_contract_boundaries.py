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

    def test_branch_join(self):
        self.check_case("both_arms", """fn probe(file: FileHandle, choose: bool) -> void {
    if choose { unsafe { (consume_handle file) } }
    else { unsafe { (consume_handle file) } }
}""", True)
        self.check_case("one_arm", """fn probe(file: FileHandle, choose: bool) -> void {
    if choose { unsafe { (consume_handle file) } }
    unsafe { (consume_handle file) }
}""", False)

    def test_early_return(self):
        self.check_case("resolved_return", """fn probe(file: FileHandle, leave: bool) -> int {
    if leave { unsafe { (consume_handle file) } return 1 }
    unsafe { (consume_handle file) }
    return 0
}""", True)
        self.check_case("leaking_return", """fn probe(file: FileHandle, leave: bool) -> int {
    if leave { return 1 }
    unsafe { (consume_handle file) }
    return 0
}""", False)

    def test_loop_outer_owner(self):
        self.check_case("outer_loop_move", """fn probe(file: FileHandle, repeat: bool) -> void {
    while repeat { unsafe { (consume_handle file) } }
    unsafe { (consume_handle file) }
}""", False)

    def test_loop_local_exits(self):
        for edge in ("break", "continue", ""):
            body = "let file: FileHandle = FileHandle { fd: 1 }\n"
            for resolved in (True, False):
                resolution = "unsafe { (consume_handle file) }\n" if resolved else ""
                self.check_case(f"loop_{edge}_{resolved}",
                                "fn probe(repeat: bool) -> void { while repeat {\n" + body + resolution + edge + "\n} }", resolved)

    def test_shadowed_ordinary_binding(self):
        self.check_case("ordinary_shadow", """fn probe(file: FileHandle, choose: bool) -> void {
    if choose { let file: int = 3 assert (== file 3) }
    unsafe { (consume_handle file) }
}""", True)

    def test_resource_assignment(self):
        for resolved in (True, False):
            resolution = "unsafe { (consume_handle file) }" if resolved else ""
            self.check_case(f"overwrite_{resolved}", """fn probe() -> void {
    let mut file: FileHandle = FileHandle { fd: 1 }
""" + resolution + """
    set file FileHandle { fd: 2 }
    unsafe { (consume_handle file) }
}""", resolved)

    def test_resource_collection_annotations(self):
        self.check_case("resource_array_parameter", """fn probe(files: array<FileHandle>) -> array<FileHandle> {
    return files
}""", False)
        self.check_case("resource_empty_array", """fn probe() -> array<FileHandle> {
    let files: array<FileHandle> = []
    return files
}""", False)
        self.check_case("extern_resource_array", "extern fn probe(files: array<FileHandle>) -> void", False)

    def test_indirect_resource_result(self):
        self.check_case("indirect_result_owner", """fn probe(factory: fn() -> FileHandle) -> FileHandle {
    return (factory)
}""", True)
        self.check_case("indirect_result_discard", """fn probe(factory: fn() -> FileHandle) -> void {
    (factory)
}""", False)

    def test_short_circuit_consumption(self):
        declaration = "extern fn consume_flag(owned: FileHandle) -> bool\n"
        self.check_case("conditional_move", declaration + """fn probe(file: FileHandle, gate: bool) -> void {
    unsafe { let condition: bool = (and gate (consume_flag file)) }
    unsafe { (consume_handle file) }
}""", False)
        self.check_case("unconditional_left_move", declaration + """fn probe(file: FileHandle, gate: bool) -> void {
    unsafe { let condition: bool = (and (consume_flag file) gate) }
}""", True)

    def test_hidden_outer_owner_still_has_obligation(self):
        self.check_case("hidden_owner", """fn probe(file: FileHandle) -> void {
    if true {
        let file: FileHandle = FileHandle { fd: 2 }
        unsafe { (consume_handle file) }
    }
}""", False)


if __name__ == "__main__":
    unittest.main()
