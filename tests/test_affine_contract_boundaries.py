"""I test ownership decisions without treating a field read as destruction."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "obj" / "test_affine_c_frontend"
PREFIX = """resource struct FileHandle { fd: int }
extern fn consume_handle(owned: FileHandle) -> void
"""
OWNERSHIP = r"(?i)(ownership|resource.{0,80}(scope|leak|live|consum)|moved value|after.{0,30}(mov|consum))"
PUBLIC_C_RESOURCE_REFUSAL = "[c_backend] I require a supported exact C value representation."
PRIOR_OUTPUT = b"prior artifact"
EXPECTED_CASE_COUNT = 36
EXPECTED_POSITIVE_COUNT = 14
EXPECTED_NEGATIVE_COUNT = 22
EXPECTED_SOURCE_SHA256 = {
    "both_arms": "5b42b7989956a7dc787ff6ed21316836da11c9281c9fa4201e6b410e75048747",
    "one_arm": "2f7985949ae4bcbb0bc2d9521c293f3fda217cdd65418bd494815aeb69fbf120",
    "resolved_return": "3abd916ef609ed8e25569e19e759bde890ce4f2ce23922eafc020559b69da0af",
    "leaking_return": "f2b71c88a675ef27cc42bde5fbaa39b2abed23db436f124e5ddaeab4d07af150",
    "observe_leak": "d04a9bb97127ee049535c22af702e5294ecfd4c04da277cbf78b99a686db8208",
    "hidden_owner": "c8f7ed506283661927f4c9b524e3551ee28cd9775feb58c9d203a0faf5389836",
    "indirect_result_owner": "744bff273790a98e95255372cc833f032146f02b40bfd5ce620fa83ee22bcf7b",
    "indirect_result_discard": "93ef171a2933e43480065f34f78ce24de3048c98a96b6225a2e990ef26e2b28d",
    "loop_break_True": "c332cae3f4f10811ca3b4cd552becc5141b06e6f2c6069dc686f368cf16d7311",
    "loop_break_False": "564c4b8a0375aa5103d1ae057813cdfe5b4ee20e949418a7ec8db311888705e0",
    "loop_continue_True": "e832166e7ce29fb1e8fede5b04a4935484e3d0a89149b79fb34b161051657d86",
    "loop_continue_False": "a73bc8725679fb769c29dbc84e8a9d2306b21b23e8a17045ceb94b8505909366",
    "loop__True": "7d9c01e6cfec6ad919b9a85621508d5923814b146aa0afc6ef01987aef913244",
    "loop__False": "dba3af1d7272a97818a53e1bc16aabbe774e8b5c0412511f075c8d8a1182eed7",
    "outer_loop_move": "70b4472263825596ddbc96e27dbbd2deea2fe263b5df6933787df421fc948c19",
    "move_consume": "2e3c34b74c1fe69daa30327917afe3bf629b55ddac2806a0ed5a362b8eb3b0f1",
    "nested_leak": "a832f4619c7b25402a1ecabdc3ecf316ccba70b704725f910e7a00bd31c57535",
    "nested_return": "95674ea51215920bcb888053ea6c36ac9962714b1019917499411d85c8dca5da",
    "nested_after_move": "b55c4067c4f3856dfd03f104ebf084d64e9efa97245ac5bdd777ba5d25208b5d",
    "nested_collection": "0341541d986d895fc41f73956a7868d16e39866088b0d5d65e3bc25682aa9dff",
    "moved_observation": "3f5b2db2a9c06679103bf2005a989ce3f139aab702ccabc996f0194f6f1195b1",
    "observe_return": "ee08e2aab8fcb164612e7b49e3a420b98d2fb03464211790069ecc629353e58e",
    "owner_257": "86f1b89e0dbcf662f0d9da8ac41e9abca3e987e0f285fe14c69fbcb4b5968909",
    "overwrite_True": "72132888182eb734703f779c997518aa26c9dceb19a5d5bbc506adba3a2093cf",
    "overwrite_False": "3a40dbfa2a06c39fe2fbcb0d79604bf9c401e43f097b81e82560a85bccf80577",
    "resource_array_parameter": "9ea9bae861ce12401306da0e15bdd242b406ebb0478a4e30ee9b9deb069cd2ab",
    "resource_empty_array": "6de3bdd7a63025b34ae5a7ba4022295f6a38139047bc1b2c1016ec4ec108539e",
    "extern_resource_array": "b45cf2fca02b2e1be80e92dd60487aec5155bc62517f30912654c7ee831874d5",
    "return_parameter": "a6b2e1e17432e95da23af725411fcad9f2ad4ceefa34cc0b6f425f317e8bf9b2",
    "ordinary_shadow": "dc82d7e545ef5b0688bd022ea1faa449db71955847140081c5ae1dd443f2f942",
    "conditional_move": "11ad67325493ef54860b746dab929b58d9c1f7cf4d4c0ae891f4a4a197fe5f51",
    "unconditional_left_move": "18f9bbd5de85dfbe4bb7ebd3354f880add864e6664dc2ced713c9a82f1c9ecbc",
    "union_leak": "4c92fea53bc8aa347113e22ddf9b65eb917125f74b919f8724f6e1518bfa4a01",
    "union_return": "dc178337029b71d09558e1e876125b55a699577d7318dbad255df9d354e35f22",
    "union_envelope_leak": "c65a9a7a936c27c25ea0abff1e5bc3a45a0008042c7ae536e3cec983eb2a5064",
    "parameter_leak": "432346590df12375364b0bb49f6a7d24bf079ab22cacd2721c0a3af98364941a",
}


def source_bytes(declaration):
    return (PREFIX + declaration + "\nfn main() -> int { return 0 }\n").encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def captured_bytes(value):
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode()


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
        negatives = len(cases) - positives
        if negatives != EXPECTED_NEGATIVE_COUNT:
            raise AssertionError(f"expected {EXPECTED_NEGATIVE_COUNT} affine negatives, found {negatives}")
        if set(names) != set(EXPECTED_SOURCE_SHA256):
            raise AssertionError("affine case names must exactly match my frozen source-hash table")
        for name, declaration, _ in cases:
            actual = sha256_bytes(source_bytes(declaration))
            expected = EXPECTED_SOURCE_SHA256[name]
            if actual != expected:
                raise AssertionError(
                    f"affine source bytes changed for {name}: expected {expected}, found {actual}"
                )
        evidence = os.environ.get("AFFINE_PROFILE_EVIDENCE_DIR")
        cls.evidence_root = Path(evidence).resolve() if evidence else None
        if cls.evidence_root is not None:
            cls.evidence_root.mkdir(parents=True, exist_ok=True)

    def capture_directory(self, name, authority):
        label = f"{name}--{authority}".replace(os.sep, "_")
        if self.evidence_root is not None:
            path = self.evidence_root / label
            path.mkdir(parents=True, exist_ok=False)
            return path, False
        return Path(tempfile.mkdtemp(prefix=f"nano-affine-observation-{label}-")), True

    def invoke_route(self, name, authority, command, source, output=None):
        capture, ephemeral = self.capture_directory(name, authority)
        source_value = source.read_bytes()
        output_before = output.read_bytes() if output is not None and output.exists() else None
        started = time.monotonic()
        terminal = "completed"
        returncode = None
        stdout = b""
        stderr = b""
        terminal_detail = None
        try:
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, timeout=120,
            )
            returncode = result.returncode
            stdout = captured_bytes(result.stdout)
            stderr = captured_bytes(result.stderr)
        except subprocess.TimeoutExpired as error:
            terminal = "timeout"
            stdout = captured_bytes(error.stdout)
            stderr = captured_bytes(error.stderr)
            terminal_detail = str(error)
        except OSError as error:
            terminal = "setup_error"
            terminal_detail = f"{type(error).__name__}: {error}"
            stderr = terminal_detail.encode()
        finally:
            elapsed = time.monotonic() - started
            output_exists = output is not None and output.exists()
            output_after = output.read_bytes() if output_exists else None
            executable = Path(command[0])
            resolved_executable = executable.resolve(strict=False)
            executable_bytes = (
                resolved_executable.read_bytes()
                if resolved_executable.is_file()
                else None
            )
            record = {
                "authority": authority,
                "case": name,
                "command": [str(argument) for argument in command],
                "elapsed_seconds": elapsed,
                "executable": str(executable),
                "executable_resolved": str(resolved_executable),
                "executable_sha256": (
                    sha256_bytes(executable_bytes) if executable_bytes is not None else None
                ),
                "output_after_exists": output_exists,
                "output_after_sha256": (
                    sha256_bytes(output_after) if output_after is not None else None
                ),
                "output_after_size": (
                    len(output_after) if output_after is not None else None
                ),
                "output_before_exists": output_before is not None,
                "output_before_sha256": (
                    sha256_bytes(output_before) if output_before is not None else None
                ),
                "output_before_size": (
                    len(output_before) if output_before is not None else None
                ),
                "returncode": returncode,
                "source_sha256": sha256_bytes(source_value),
                "source_size": len(source_value),
                "stderr_sha256": sha256_bytes(stderr),
                "stderr_size": len(stderr),
                "stdout_sha256": sha256_bytes(stdout),
                "stdout_size": len(stdout),
                "terminal": terminal,
                "terminal_detail": terminal_detail,
                "timeout_seconds": 120,
            }
            (capture / "source.nano").write_bytes(source_value)
            (capture / "stdout.bin").write_bytes(stdout)
            (capture / "stderr.bin").write_bytes(stderr)
            if output_before is not None:
                (capture / "output.before.bin").write_bytes(output_before)
            if output_after is not None:
                (capture / "output.after.bin").write_bytes(output_after)
            (capture / "observation.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n"
            )
        return {
            "capture": capture,
            "diagnostic": (stdout + stderr).decode(errors="replace"),
            "ephemeral": ephemeral,
            "output_after": output_after,
            "output_before": output_before,
            "returncode": returncode,
            "terminal": terminal,
        }

    def finish_observation(self, observation, passed):
        if not observation["ephemeral"]:
            return
        if passed:
            shutil.rmtree(observation["capture"])
        else:
            print(
                f"retained affine route observation: {observation['capture']}",
                file=sys.stderr,
            )

    def assert_completed(self, observation):
        self.assertEqual(
            observation["terminal"], "completed",
            f"route ended as {observation['terminal']}; evidence: {observation['capture']}",
        )

    def assert_refusal_preserved_output(self, observation):
        if observation["terminal"] == "completed" and observation["returncode"] != 0:
            self.assertEqual(observation["output_before"], PRIOR_OUTPUT)
            self.assertEqual(observation["output_after"], PRIOR_OUTPUT)

    def run_frontend(self, name, source, accepted):
        observation = self.invoke_route(
            name, "c_frontend", [str(FRONTEND), str(source)], source,
        )
        passed = False
        try:
            self.assert_completed(observation)
            diagnostic = observation["diagnostic"]
            if accepted:
                self.assertEqual(observation["returncode"], 0, diagnostic)
                self.assertIn("AFFINE_C_FRONTEND_ACCEPTED", diagnostic)
                self.assertNotIn("AFFINE_C_FRONTEND_REFUSED", diagnostic)
            else:
                self.assertEqual(observation["returncode"], 1, diagnostic)
                self.assertIn("AFFINE_C_FRONTEND_REFUSED:typecheck", diagnostic)
                self.assertRegex(diagnostic, OWNERSHIP)
            self.assertNotIn("AFFINE_C_FRONTEND_FAILED:", diagnostic, name)
            passed = True
        finally:
            self.finish_observation(observation, passed)

    def run_public_c(self, name, source, output, accepted):
        output.write_bytes(PRIOR_OUTPUT)
        observation = self.invoke_route(
            name, "public_c",
            [str(ROOT / "bin" / "nanoc_c"), str(source), "--target", "c", "-o", str(output)],
            source, output,
        )
        passed = False
        try:
            self.assert_refusal_preserved_output(observation)
            self.assert_completed(observation)
            diagnostic = observation["diagnostic"]
            self.assertGreater(observation["returncode"], 0, diagnostic)
            if accepted:
                self.assertIn(PUBLIC_C_RESOURCE_REFUSAL, diagnostic)
            else:
                self.assertRegex(diagnostic, OWNERSHIP)
            passed = True
        finally:
            self.finish_observation(observation, passed)

    def run_selfhost_c(self, name, compiler, source, output, accepted):
        output.write_bytes(PRIOR_OUTPUT)
        observation = self.invoke_route(
            name, f"selfhost_c--{compiler}",
            [str(ROOT / "bin" / compiler), str(source), "--target", "c", "-o", str(output)],
            source, output,
        )
        passed = False
        try:
            self.assert_refusal_preserved_output(observation)
            self.assert_completed(observation)
            diagnostic = observation["diagnostic"]
            if accepted:
                self.assertEqual(observation["returncode"], 0, diagnostic)
                self.assertNotEqual(observation["output_after"], PRIOR_OUTPUT)
            else:
                self.assertGreater(observation["returncode"], 0, diagnostic)
                self.assertRegex(diagnostic, OWNERSHIP)
            passed = True
        finally:
            self.finish_observation(observation, passed)

    def check_case(self, name, declaration, accepted):
        with tempfile.TemporaryDirectory(prefix="nano-affine-contract-") as directory:
            source = Path(directory) / "case.nano"
            exact_source = source_bytes(declaration)
            source.write_bytes(exact_source)
            self.assertEqual(source.read_bytes(), exact_source)
            self.assertEqual(sha256_bytes(exact_source), EXPECTED_SOURCE_SHA256[name])

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
