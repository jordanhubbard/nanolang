"""I characterize implementation boundaries; these are not universal language proofs."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = {
    "c-seed": ROOT / "bin/nanoc_c",
    "selfhost": Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve(),
    "bytecode": ROOT / "bin/nano_virt",
}


class LanguageClaims(unittest.TestCase):
    def test_developer_document_example(self):
        with ZipFile(ROOT / "docs/presentation/nanolang-developer-overview.docx") as archive:
            document = ET.fromstring(archive.read("word/document.xml"))
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = ["".join("\n" if node.tag.endswith("}br") else node.text or ""
                              for node in paragraph.iter()
                              if node.tag in (f"{{{namespace['w']}}}t", f"{{{namespace['w']}}}br"))
                      for paragraph in document.findall(".//w:p", namespace)]
        examples = [value for value in paragraphs if value.startswith("fn gcd(")]
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example, (ROOT / "docs/presentation/examples/gcd.nano").read_text().strip())
        source = example + "\nfn main() -> int { return (gcd 48 18) }\nshadow main { assert (== (main) 6) }\n"
        for backend in COMPILERS:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory(prefix="nano-document-code-") as tmp:
                compiled, output = self.compile_source(backend, source, Path(tmp))
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                self.assertEqual(self.execute(backend, output).returncode, 6)
                previous = output.read_bytes()
                broken = source.replace("(gcd 48 18) 6", "(gcd 48 18) 7")
                rejected, output = self.compile_source(backend, broken, Path(tmp))
                self.assertGreater(rejected.returncode, 0, rejected.stdout + rejected.stderr)
                self.assertEqual(output.read_bytes(), previous)

    def test_canonical_import_paths(self):
        for backend in COMPILERS:
            for duplicate in (False, True):
                with self.subTest(backend=backend, duplicate=duplicate), tempfile.TemporaryDirectory(prefix="nano-canonical-import-") as tmp:
                    directory = Path(tmp)
                    target = directory / "target"
                    target.mkdir()
                    (target / "helper.nano").write_text("pub fn answer() -> int { return 41 }\nshadow answer { assert (== (answer) 41) }\n")
                    leaf = target / "leaf.nano"
                    leaf.write_text('module "helper.nano" as dep\npub fn answer() -> int { return (+ (dep.answer) 1) }\n'
                                    'shadow answer { assert (== (answer) 42) }\n')
                    link = directory / "link.nano"
                    link.symlink_to(leaf)
                    prefix = f'module "{leaf}" as original\n' if duplicate else ""
                    source = prefix + f'module "{link}" as lib\nfn main() -> int {{ return (lib.answer) }}\nshadow main {{ assert (== (main) 42) }}\n'
                    compiled, output = self.compile_source(backend, source, directory)
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    executed = self.execute(backend, output)
                    self.assertEqual(executed.returncode, 42, executed.stdout + executed.stderr)

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
                    self.assertNotEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertFalse(output.exists())
                    compiled, output = self.compile_source(backend, source, directory, ("--root-shadows-only",))
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

    def test_qualified_same_named_wrapper(self):
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
                self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                result = self.execute(backend, output)
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[:4000])

    def test_local_duplicate_function_is_rejected(self):
        for backend in COMPILERS:
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

    def test_selfhost_duplicate_owner_preserves_output(self):
        for depth, emit_c in ((depth, emit_c) for depth in (0, 1, 2) for emit_c in (False, True)):
            with self.subTest(depth=depth, emit_c=emit_c), tempfile.TemporaryDirectory(prefix="nano-owner-duplicate-") as tmp:
                directory = Path(tmp)
                body = "fn duplicate() -> int { return 1 }\nfn duplicate() -> int { return 2 }\n"
                for level in range(depth):
                    target = directory / f"dependency{level}.nano"
                    target.write_text(body)
                    body = f'module "{target}" as dependency{level}\n'
                output = directory / "claim"
                output.write_bytes(b"previous executable")
                report = directory / "diagnostics.json"
                args = ["--llm-diags-json", str(report)]
                if emit_c:
                    args.extend(["--target", "c"])
                compiled, actual = self.compile_source("selfhost", body + "fn main() -> int { return 0 }\nshadow main { assert true }\n", directory, args)
                self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                self.assertIn(b"twice in one module", compiled.stdout + compiled.stderr)
                self.assertEqual(actual.read_bytes(), b"previous executable")
                diagnostic = json.loads(report.read_text())
                self.assertEqual(diagnostic["exit_code"], 1)
                self.assertIn("M0001", [item["code"] for item in diagnostic["diagnostics"]])

    def test_selfhost_repeated_extern_declaration(self):
        source = "extern fn erf(x: float) -> float\nextern fn erf(x: float) -> float\nfn main() -> int { return 0 }\nshadow main { assert true }\n"
        with tempfile.TemporaryDirectory(prefix="nano-owner-extern-") as tmp:
            compiled, output = self.compile_source("selfhost", source, Path(tmp))
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            self.assertEqual(self.execute("selfhost", output).returncode, 0)

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

    def test_aliases_belong_to_the_importing_module(self):
        for backend in COMPILERS:
            for declared in (False, True):
                with self.subTest(backend=backend, declared=declared), tempfile.TemporaryDirectory(prefix="nano-alias-owners-") as tmp:
                    directory = Path(tmp)
                    for name, number in (("left", 11), ("right", 22)):
                        leaf = directory / f"{name}_value.nano"
                        owner = f"module {name.title()}Value\n" if declared else ""
                        leaf.write_text(owner + f"pub fn answer() -> int {{ return {number} }}\nshadow answer {{ assert (== (answer) {number}) }}\n")
                        wrapper = directory / f"{name}_wrapper.nano"
                        owner = f"module {name.title()}Wrapper\n" if declared else ""
                        wrapper.write_text(owner + f'''module "{leaf}" as lib
pub fn answer() -> int {{ return (lib.answer) }}
shadow answer {{ assert (== (answer) {number}) }}
''')
                    source = f'''module "{directory}/left_wrapper.nano" as left
module "{directory}/right_wrapper.nano" as right
fn main() -> int {{ assert (== (left.answer) 11) assert (== (right.answer) 22) return 0 }}
shadow main {{ assert (== (main) 0) }}
'''
                    compiled, output = self.compile_source(backend, source, directory)
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    executed = self.execute(backend, output)
                    self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)

    def test_qualified_and_returned_foreign_dispatch(self):
        for route in ("qualified", "returned", "variable", "map"):
            for missing in (False, True):
                with self.subTest(route=route, missing=missing), tempfile.TemporaryDirectory(prefix="nano-foreign-route-") as tmp:
                    directory = Path(tmp)
                    symbol = "nano_missing_route_symbol" if missing else "erf"
                    declaration = f"extern fn {symbol}(x: float) -> float\n"
                    if route == "qualified":
                        library = directory / "foreign.nano"
                        library.write_text("pub " + declaration)
                        prefix = f'unsafe module "{library}" as foreign\n'
                        call = f"(foreign.{symbol} 0.0)"
                    elif route == "returned":
                        prefix = declaration + f'''fn choose() -> fn(float) -> float {{ return {symbol} }}
shadow choose {{ unsafe {{ assert (== ((choose) 0.0) 0.0) }} }}
'''
                        call = "((choose) 0.0)"
                    else:
                        prefix = declaration
                        call = f"(map [0.0] {symbol})" if route == "map" else "(op 0.0)"
                    setup = f"let op: fn(float) -> float = {symbol} " if route == "variable" else ""
                    value = f"(array_get {call} 0)" if route == "map" else call
                    operation = call if missing else f"assert (== {value} 0.0)"
                    source = prefix + f'''fn root() -> int {{ return 42 }}
shadow root {{ unsafe {{ {setup}{operation} }} assert (== (root) 42) }}
fn main() -> int {{ return 0 }}
shadow main {{ assert (== (main) 0) }}
'''
                    report = directory / "shadows.json"
                    output = directory / "claim"
                    output.write_bytes(b"prior artifact")
                    compiled, output = self.compile_source("c-seed", source, directory,
                                                           ["--llm-shadow-json", str(report)])
                    self.assertTrue(report.exists(), compiled.stdout + compiled.stderr)
                    evidence = json.loads(report.read_text())
                    if missing:
                        self.assertFalse(evidence["success"], compiled.stdout + compiled.stderr)
                        failure = next(f for f in evidence["failures"] if f["test"] == "root")
                        self.assertEqual(failure["fail_count"], 1)
                        lines = source.splitlines()
                        line = next(i for i, value in enumerate(lines, 1) if value.startswith("shadow root"))
                        self.assertEqual(failure["first_location"]["line"], line)
                        self.assertGreater(failure["first_location"]["column"], 0)
                        self.assertNotEqual(compiled.returncode, 0)
                        self.assertEqual(output.read_bytes(), b"prior artifact")
                    else:
                        self.assertTrue(evidence["success"], compiled.stdout + compiled.stderr)
                        self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                        self.assertEqual(self.execute("c-seed", output).returncode, 0)

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
