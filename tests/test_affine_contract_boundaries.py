"""I test ownership decisions without treating a field read as destruction."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "obj" / "test_affine_c_frontend"
PREFIX = """resource struct FileHandle { fd: int }
extern fn consume_handle(owned: FileHandle) -> void
"""
OWNERSHIP = r"(?i)(ownership|resource.{0,80}(scope|leak|live|consum)|moved value|after.{0,30}(mov|consum))"
PUBLIC_C_PROFILE_REFUSALS = {
    "indirect_result_owner": "I do not provide first-class callable values or indirect calls in this C profile.",
    "union_return": "I require scalar C union result payloads.",
}
EXPECTED_CASE_COUNT = 36
EXPECTED_POSITIVE_COUNT = 14


class AffineContractBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = collect_affine_contract_cases()
        names = [name for name, _, _ in cases]
        if len(cases) != EXPECTED_CASE_COUNT:
            raise AssertionError(f"expected {EXPECTED_CASE_COUNT} affine cases, found {len(cases)}")
        if len(set(names)) != len(names):
            raise AssertionError("affine case names must be unique")
        positives = sum(accepted for _, _, accepted in cases)
        if positives != EXPECTED_POSITIVE_COUNT:
            raise AssertionError(f"expected {EXPECTED_POSITIVE_COUNT} affine positives, found {positives}")
        if set(PUBLIC_C_PROFILE_REFUSALS) != {name for name, _, accepted in cases if accepted and name in PUBLIC_C_PROFILE_REFUSALS}:
            raise AssertionError("public C profile refusals must name existing semantic positives")

    def run_frontend(self, name, source, accepted):
        result = subprocess.run(
            [str(FRONTEND), str(source)], cwd=ROOT,
            capture_output=True, text=True, timeout=120,
        )
        diagnostic = result.stdout + result.stderr
        if accepted:
            self.assertEqual(result.returncode, 0, diagnostic)
            self.assertIn("AFFINE_C_FRONTEND_ACCEPTED", diagnostic)
            self.assertNotIn("AFFINE_C_FRONTEND_REFUSED", diagnostic)
        else:
            self.assertEqual(result.returncode, 1, diagnostic)
            self.assertIn("AFFINE_C_FRONTEND_REFUSED:typecheck", diagnostic)
            self.assertRegex(diagnostic, OWNERSHIP)
        self.assertNotIn("AFFINE_C_FRONTEND_FAILED:", diagnostic, name)

    def run_public_c(self, name, source, output, accepted):
        output.write_bytes(b"prior artifact")
        result = subprocess.run(
            [str(ROOT / "bin" / "nanoc_c"), str(source), "--target", "c", "-o", str(output)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        diagnostic = result.stdout + result.stderr
        refusal = PUBLIC_C_PROFILE_REFUSALS.get(name)
        if refusal is not None:
            self.assertTrue(accepted, f"{name} must remain a semantic positive")
            self.assertGreater(result.returncode, 0, diagnostic)
            self.assertIn(refusal, diagnostic)
            self.assertEqual(output.read_bytes(), b"prior artifact")
        elif accepted:
            self.assertEqual(result.returncode, 0, diagnostic)
            self.assertNotEqual(output.read_bytes(), b"prior artifact")
        else:
            self.assertGreater(result.returncode, 0, diagnostic)
            self.assertRegex(diagnostic, OWNERSHIP)
            self.assertEqual(output.read_bytes(), b"prior artifact")

    def run_selfhost_c(self, name, compiler, source, output, accepted):
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

    def check_case(self, name, declaration, accepted):
        with tempfile.TemporaryDirectory(prefix="nano-affine-contract-") as directory:
            source = Path(directory) / "case.nano"
            source_bytes = (PREFIX + declaration + "\nfn main() -> int { return 0 }\n").encode()
            source.write_bytes(source_bytes)
            self.assertEqual(source.read_bytes(), source_bytes)

            with self.subTest(case=name, authority="c_frontend"):
                self.run_frontend(name, source, accepted)
            with self.subTest(case=name, authority="public_c"):
                self.run_public_c(name, source, Path(directory) / "public.c", accepted)
            for compiler in ("nanoc_stage1", "nanoc_stage2"):
                with self.subTest(case=name, authority="selfhost_c", compiler=compiler):
                    self.run_selfhost_c(
                        name, compiler, source, Path(directory) / f"{compiler}.c", accepted
                    )

    def test_return_owned_parameter(self):
        self.check_case("return_parameter", "fn probe(file: FileHandle) -> FileHandle { return file }", True)

    def test_nested_record_parameter_obligations(self):
        types = "struct Inner { file: FileHandle }\nstruct Outer { inner: Inner }\n"
        self.check_case("nested_leak", types + "fn probe(owner: Outer) -> int { return 0 }", False)
        self.check_case("nested_return", types + "fn probe(owner: Outer) -> Outer { return owner }", True)
        self.check_case("nested_after_move", types + """extern fn consume_outer(owner: Outer) -> void
fn probe(owner: Outer) -> void {
    let next: Outer = owner
    unsafe { (consume_outer next) (consume_outer owner) }
}""", False)

    def test_union_payload_parameter_obligations(self):
        types = "union Choice { Some { file: FileHandle }, None {} }\nstruct Envelope { choice: Choice }\n"
        self.check_case("union_leak", types + "fn probe(owner: Choice) -> int { return 0 }", False)
        self.check_case("union_return", types + "fn probe(owner: Choice) -> Choice { return owner }", True)
        self.check_case("union_envelope_leak", types + "fn probe(owner: Envelope) -> int { return 0 }", False)

    def test_nested_resource_collection_signature(self):
        self.check_case("nested_collection", """struct Box { file: FileHandle }
extern fn unsupported(items: array<Box>) -> void
""", False)

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


def collect_affine_contract_cases():
    cases = []
    original = AffineContractBoundaries()
    original.check_case = lambda name, declaration, accepted: cases.append(
        (name, declaration, accepted)
    )
    methods = sorted(
        name for name in vars(AffineContractBoundaries) if name.startswith("test_")
    )
    for method in methods:
        getattr(original, method)()
    return cases


if __name__ == "__main__":
    unittest.main()
