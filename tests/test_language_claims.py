"""I characterize implementation boundaries; these are not universal language proofs."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = {
    "c-seed": ROOT / "bin/nanoc_c",
    "selfhost": Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve(),
    "bytecode": ROOT / "bin/nano_virt",
}


class LanguageClaims(unittest.TestCase):
    def compile_source(self, backend, source, directory, extra_args=()):
        path = directory / "claim.nano"
        path.write_text(source)
        output = directory / ("claim.nvm" if backend == "bytecode" else "claim")
        args = [str(COMPILERS[backend]), str(path), "-o", str(output)]
        if backend == "bytecode":
            args.append("--emit-nvm")
        args.extend(extra_args)
        compiled = subprocess.run(args, cwd=ROOT, capture_output=True,
                                  env=dict(os.environ, TMPDIR=str(directory)), timeout=60)
        return compiled, output

    def execute(self, backend, output):
        args = [str(ROOT / "bin/nano_vm"), str(output)] if backend == "bytecode" else [str(output)]
        return subprocess.run(args, cwd=ROOT, capture_output=True, timeout=10)

    def test_local_inference_and_grouping(self):
        source = '''fn main() -> int {
    let x = 2 + 3 * 4
    let y = 2 + (3 * 4)
    let name = "typed"
    assert (== x 20)
    assert (== y 14)
    assert (== (str_length name) 5)
    return 0
}
shadow main { assert (== (main) 0) }
'''
        for backend in COMPILERS:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory(prefix="nano-claims-") as tmp:
                compiled, output = self.compile_source(backend, source, Path(tmp))
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                executed = self.execute(backend, output)
                self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

    def test_rejected_type_boundaries(self):
        cases = {
            "untyped_parameter": "fn f(x) -> int { return 0 }\nshadow f { assert true }\n",
            "untyped_return": "fn f(x: int) { return x }\nshadow f { assert true }\n",
            "immutable_assignment": "fn f() -> int { let x = 1 set x 2 return x }\nshadow f { assert true }\n",
            "integer_condition": "fn f() -> int { if 1 { return 1 } return 0 }\nshadow f { assert true }\n",
        }
        for name, body in cases.items():
            for backend in COMPILERS:
                with self.subTest(case=name, backend=backend), tempfile.TemporaryDirectory(prefix="nano-claims-") as tmp:
                    source = body + "fn main() -> int { return 0 }\nshadow main { assert true }\n"
                    compiled, output = self.compile_source(backend, source, Path(tmp))
                    self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertFalse(output.exists(), "I must not publish an executable after rejecting source")

    def test_missing_shadow_is_not_universally_rejected(self):
        source = "fn f() -> int { return 42 }\nfn main() -> int { assert (== (f) 42) return 0 }\n"
        for backend in COMPILERS:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory(prefix="nano-claims-") as tmp:
                compiled, output = self.compile_source(backend, source, Path(tmp))
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                self.assertEqual(self.execute(backend, output).returncode, 0)

    def test_shadow_execution_rejects_failure(self):
        source = "fn f() -> int { return 42 }\nshadow f { assert false }\nfn main() -> int { return 0 }\nshadow main { assert true }\n"
        for backend in COMPILERS:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory(prefix="nano-claims-") as tmp:
                compiled, output = self.compile_source(backend, source, Path(tmp))
                self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                self.assertIn(b"shadow", (compiled.stdout + compiled.stderr).lower())
                self.assertFalse(output.exists())

    def test_specification_unsafe_math_example(self):
        specification = (ROOT / "docs/SPECIFICATION.md").read_text()
        section = specification.split("### 6.4 External Functions (FFI)", 1)[1].split("## 7.", 1)[0]
        examples = re.findall(r"```nano\n(.*?)```", section, flags=re.S)
        example = next(code for code in examples if "fn hypotenuse" in code)
        source = example + "\nfn main() -> int { assert (== (hypotenuse 3.0 4.0) 5.0) return 0 }\nshadow main { assert true }\n"
        with tempfile.TemporaryDirectory(prefix="nano-claims-") as tmp:
            compiled, output = self.compile_source("c-seed", source, Path(tmp))
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            self.assertEqual(self.execute("c-seed", output).returncode, 0)

    def test_imported_shadow_selection(self):
        dependency = '''pub fn answer() -> int { return 42 }
shadow answer { assert false }
'''
        for backend in COMPILERS:
            for transitive in (False, True):
                with self.subTest(backend=backend, transitive=transitive), tempfile.TemporaryDirectory(prefix="nano-import-shadows-") as tmp:
                    directory = Path(tmp)
                    leaf = directory / "leaf.nano"
                    leaf.write_text(dependency)
                    imported = leaf
                    function = "answer"
                    if transitive:
                        imported = directory / "middle.nano"
                        function = "forward_answer"
                        imported.write_text(f'''module "{leaf}" as leaf
pub fn forward_answer() -> int {{ return (leaf.answer) }}
shadow forward_answer {{ assert (== (forward_answer) 42) }}
''')
                    source = f'''module "{imported}" as helper
fn main() -> int {{ assert (== (helper.{function}) 42) return 0 }}
shadow main {{ assert (== (helper.{function}) 42) }}
'''
                    compiled, output = self.compile_source(backend, source, directory)
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    executed = self.execute(backend, output)
                    self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

                    root_directory = directory / "root-control"
                    root_directory.mkdir()
                    compiled, output = self.compile_source(
                        backend, dependency + "fn main() -> int { return 0 }\nshadow main { assert true }\n",
                        root_directory)
                    self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertIn(b"shadow", (compiled.stdout + compiled.stderr).lower())
                    self.assertFalse(output.exists())

    def test_qualified_same_named_wrapper_backend_boundary(self):
        for backend, declared in ((backend, declared) for backend in COMPILERS for declared in (False, True)):
            with self.subTest(backend=backend, declared=declared), tempfile.TemporaryDirectory(prefix="nano-owner-") as tmp:
                directory = Path(tmp)
                leaf = directory / "leaf.nano"
                leaf.write_text(("module LeafOwner\n" if declared else "") + "fn increment() -> int { return 40 }\nshadow increment { assert (== (increment) 40) }\npub fn answer() -> int { return (increment) }\nshadow answer { assert (== (answer) 40) }\n")
                middle = directory / "middle.nano"
                middle.write_text(("module MiddleOwner\n" if declared else "") + f'''module "{leaf}" as leaf
fn increment() -> int {{ return 2 }}
shadow increment {{ assert (== (increment) 2) }}
pub fn answer() -> int {{ return (+ (leaf.answer) (increment)) }}
shadow answer {{ assert (== (answer) 42) }}
''')
                source = f'''module "{middle}" as middle
fn increment() -> int {{ return 3 }}
shadow increment {{ assert (== (increment) 3) }}
fn answer() -> int {{ return (+ (middle.answer) (increment)) }}
shadow answer {{ assert (== (answer) 45) }}
fn main() -> int {{ assert (== (answer) 45) assert (== (middle.answer) 42) return 0 }}
shadow main {{ assert (== (main) 0) }}
'''
                compiled, output = self.compile_source(backend, source, directory)
                if backend == "selfhost":
                    # I still flatten imports in Stage2. This useful root shadow
                    # catches the wrong implementation before output publication.
                    self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertIn(b"failed shadow", compiled.stdout + compiled.stderr)
                    self.assertFalse(output.exists())
                    continue
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                result = self.execute(backend, output)
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[:4000])

    def test_local_duplicate_function_is_rejected(self):
        for backend in ("c-seed", "bytecode"):
            for imported in (False, True):
                with self.subTest(backend=backend, imported=imported), tempfile.TemporaryDirectory(prefix="nano-duplicate-") as tmp:
                    directory = Path(tmp)
                    body = "fn answer() -> int { return 1 }\nfn answer() -> int { return 2 }\nshadow answer { assert true }\n"
                    if imported:
                        dependency = directory / "duplicate.nano"
                        dependency.write_text(body)
                        body = f'module "{dependency}" as duplicate\n'
                    compiled, output = self.compile_source(backend, body + "fn main() -> int { return 0 }\nshadow main { assert true }\n", directory)
                    self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertFalse(output.exists())

    def test_failed_import_preserves_output(self):
        for backend in ("c-seed", "bytecode"):
            for failure in ("duplicate", "type", "syntax", "missing", "cycle"):
                for transitive in (False, True):
                    with self.subTest(backend=backend, failure=failure, transitive=transitive), tempfile.TemporaryDirectory(prefix="nano-import-error-") as tmp:
                        directory = Path(tmp)
                        leaf = directory / "leaf.nano"
                        if failure == "duplicate":
                            leaf.write_text("fn answer() -> int { return 1 }\nfn answer() -> int { return 2 }\n")
                        elif failure == "type":
                            leaf.write_text('fn answer() -> int { return "wrong" }\n')
                        elif failure == "syntax":
                            leaf.write_text("fn answer(\n")
                        elif failure == "cycle":
                            leaf.write_text(f'module "{leaf}" as cycle\n')
                        target = leaf
                        if transitive:
                            target = directory / "middle.nano"
                            target.write_text(f'module "{leaf}" as leaf\n')
                        output = directory / ("claim.nvm" if backend == "bytecode" else "claim")
                        output.write_bytes(b"previous output")
                        source = f'module "{target}" as dependency\nfn main() -> int {{ return 0 }}\nshadow main {{ assert true }}\n'
                        compiled, actual = self.compile_source(backend, source, directory)
                        self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                        self.assertEqual(actual.read_bytes(), b"previous output")

    def test_foreign_function_does_not_exempt_explicit_shadow(self):
        for body in ("if false { return (erf 0.0) } return 0.0",
                     "unsafe { if false { return (erf 0.0) } } return 0.0",
                     "return 0.0"):
            for passed in (False, True):
                with self.subTest(body=body, passed=passed), tempfile.TemporaryDirectory(prefix="nano-foreign-shadow-") as tmp:
                    shadow_prefix = "if false { let unused: float = (erf 0.0) }" if body == "return 0.0" else ""
                    source = f'''extern fn erf(x: float) -> float
fn root() -> float {{ {body} }}
shadow root {{ {shadow_prefix} assert (== (root) {0.0 if passed else 3.0}) }}
fn main() -> int {{ return 0 }}
shadow main {{ assert (== (main) 0) }}
'''
                    compiled, output = self.compile_source("c-seed", source, Path(tmp))
                    if passed:
                        self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                        self.assertEqual(self.execute("c-seed", output).returncode, 0)
                    else:
                        self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                        self.assertIn(b"shadow", (compiled.stdout + compiled.stderr).lower())
                        self.assertFalse(output.exists())

    def test_foreign_failures_reject_ignored_results(self):
        # I match the host's int64_t typedef in the generated C declaration.
        absolute = "llabs" if sys.platform == "darwin" else "labs"
        cases = {
            "missing": ("extern fn nano_missing_shadow_symbol() -> void", "(nano_missing_shadow_symbol)"),
            "float": ("extern fn erf(x: float) -> float", "assert (== (erf 0.0) 0.0)"),
            "float_failure": ("extern fn erf(x: float) -> float", "assert (== (erf 0.0) 1.0)"),
            "resolved": (f"extern fn {absolute}(x: int) -> int", f"assert (== ({absolute} -42) 42)"),
        }
        for name, (declaration, call) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix="nano-foreign-failure-") as tmp:
                directory = Path(tmp)
                output = directory / "claim"
                output.write_bytes(b"previous artifact")
                source = f'''{declaration}
fn root() -> int {{ return 42 }}
shadow root {{ unsafe {{ {call} }} assert (== (root) 42) }}
fn main() -> int {{ return 0 }}
shadow main {{ assert (== (main) 0) }}
'''
                report = directory / "shadows.json"
                compiled, output = self.compile_source("c-seed", source, directory,
                                                       ["--llm-shadow-json", str(report)])
                self.assertTrue(report.exists(), compiled.stdout + compiled.stderr)
                evidence = json.loads(report.read_text())
                if name in ("resolved", "float"):
                    self.assertTrue(evidence["success"])
                    self.assertEqual(evidence["failures"], [])
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertEqual(self.execute("c-seed", output).returncode, 0)
                else:
                    self.assertFalse(evidence["success"])
                    self.assertEqual(len(evidence["failures"]), 1)
                    failure = evidence["failures"][0]
                    self.assertEqual(failure["test"], "root")
                    self.assertEqual(failure["fail_count"], 1)
                    self.assertEqual(failure["first_location"]["line"], 3)
                    self.assertGreater(failure["first_location"]["column"], 0)
                    self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertIn(b"Shadow tests failed", compiled.stdout + compiled.stderr)
                    self.assertEqual(output.read_bytes(), b"previous artifact")


if __name__ == "__main__":
    unittest.main()
