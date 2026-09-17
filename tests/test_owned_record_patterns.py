"""I require complete owned patterns and execute their transferred bindings."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = """resource struct Handle { fd: int }
fn close(owned: Handle) -> int { let Handle { fd } = owned return fd }
shadow close { assert (== (close Handle { fd: 7 }) 7) }
"""


class OwnedRecordPatterns(unittest.TestCase):
    def check_case(self, source, accepted, stdout=None, diagnostic=None, prefix=PREFIX):
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2", "nano_virt"):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-owned-pattern-") as tmp:
                path = Path(tmp) / "case.nano"
                output = Path(tmp) / ("case.nvm" if compiler == "nano_virt" else "program")
                path.write_text(prefix + source)
                output.write_bytes(b"prior artifact")
                command = [str(ROOT / "bin" / compiler), str(path), "-o", str(output)]
                if compiler == "nano_virt":
                    command.append("--emit-nvm")
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
                if accepted:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    run = [str(output)] if compiler != "nano_virt" else [str(ROOT / "bin/nano_vm"), str(output)]
                    result = subprocess.run(run, cwd=ROOT, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    if stdout is not None:
                        self.assertEqual(result.stdout, stdout)
                else:
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    if diagnostic is not None:
                        self.assertRegex(result.stdout + result.stderr, diagnostic)
                    self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_scalar_terminal_operation(self):
        self.check_case("""fn main() -> int {
    let first: Handle = Handle { fd: 42 }
    let next: Handle = first
    assert (== (close next) 42)
    return 0
}
shadow main { assert (== (main) 0) }
""", True)

    def test_nested_transfer_and_unsafe_scope(self):
        self.check_case("""struct Bundle { file: Handle, label: string }
fn unpack(bundle: Bundle) -> int {
    unsafe { let Bundle { label, file } = bundle
        assert (== label "ready")
        return (close file)
    }
}
shadow unpack { assert (== (unpack Bundle { file: Handle { fd: 19 }, label: "ready" }) 19) }
fn main() -> int {
    assert (== (unpack Bundle { file: Handle { fd: 19 }, label: "ready" }) 19)
    return 0
}
shadow main { assert (== (main) 0) }
""", True)

    def test_invalid_patterns(self):
        for pattern in ("Handle {}", "Handle { fd, fd }", "Handle { missing }", "Handle { .. }", "Other { fd }"):
            with self.subTest(pattern=pattern):
                self.check_case("struct Other { fd: int }\nfn bad(owned: Handle) -> int { let " + pattern +
                                " = owned return 0 }\nfn main() -> int { return 0 }", False)
        self.check_case("""struct Pair { left: int, right: int }
fn bad(pair: Pair) -> int { let Pair { left, left } = pair return left }
fn main() -> int { return 0 }
""", False)

    def test_initializer_evaluated_once(self):
        self.check_case("""fn acquire() -> Handle { (println "acquired") return Handle { fd: 8 } }
shadow acquire { assert (== (close (acquire)) 8) }
fn main() -> int { let Handle { fd } = (acquire) assert (== fd 8) return 0 }
shadow main { assert (== (main) 0) }
""", True, stdout="acquired\n")

    def test_unsafe_pattern_keeps_outer_shadow(self):
        self.check_case("""fn main() -> int {
    let fd: float = 2.5
    unsafe { let Handle { fd } = Handle { fd: 42 } assert (== fd 42) }
    assert (== fd 2.5)
    return 0
}
shadow main { assert (== (main) 0) }
""", True)

    def test_ordinary_array_field_keeps_element_type(self):
        self.check_case("""struct Bundle { file: Handle, samples: array<float> }
fn main() -> int {
    let bundle: Bundle = Bundle { file: Handle { fd: 7 }, samples: [1.5, 2.5] }
    let Bundle { samples, file } = bundle
    assert (== (at samples 0) 1.5)
    assert (== (close file) 7)
    return 0
}
shadow main { assert (== (main) 0) }
""", True)

    def test_empty_resource_pattern(self):
        self.check_case("""resource struct Empty {}
fn main() -> int { let value: Empty = Empty {} let Empty {} = value return 0 }
shadow main { assert (== (main) 0) }
""", True)

    def test_ordinary_inferred_field_and_alias(self):
        self.check_case("""struct Samples { values: array<float> }
fn main() -> int {
    let record: Samples = Samples { values: [1.5, 2.5] }
    let values = record.values
    let alias = values
    assert (== (at alias 1) 2.5)
    return 0
}
shadow main { assert (== (main) 0) }
""", True)

    def test_existing_resource_examples(self):
        for filename in ("tests/test_resource_tracking.nano", "tests/test_affine_integration.nano",
                         "examples/language/nl_affine_resource_demo.nano"):
            with self.subTest(filename=filename):
                self.check_case((ROOT / filename).read_text(), True, prefix="")

    def test_source_unavailable_after_pattern(self):
        self.check_case("""fn bad(owned: Handle) -> int {
    let Handle { fd } = owned
    return owned.fd
}
fn main() -> int { return 0 }
""", False, diagnostic=r"(?i)(ownership|moved|consumed)")

    def test_transferred_field_must_be_resolved(self):
        self.check_case("""struct Bundle { file: Handle }
fn bad(bundle: Bundle) -> int { let Bundle { file } = bundle return 0 }
fn main() -> int { return 0 }
""", False, diagnostic=r"(?i)(ownership|resource.*(live|scope|consum))")

    def test_partial_move_is_not_destruction(self):
        self.check_case("""struct Bundle { file: Handle }
fn bad(bundle: Bundle) -> int {
    let alias: Handle = bundle.file
    let Bundle { file } = bundle
    return (+ (close alias) (close file))
}
fn main() -> int { return 0 }
""", False, diagnostic=r"(?i)(ownership|partially move|resource field)")


if __name__ == "__main__":
    unittest.main()
