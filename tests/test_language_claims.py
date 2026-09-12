"""I characterize implementation boundaries; these are not universal language proofs."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = {
    "c-seed": ROOT / "bin/nanoc_c",
    "selfhost": Path(os.environ.get("NANOLANG_SELFHOST_COMPILER", ROOT / "bin/nanoc_stage2")).resolve(),
    "bytecode": ROOT / "bin/nano_virt",
}


class LanguageClaims(unittest.TestCase):
    def compile_source(self, backend, source, directory):
        path = directory / "claim.nano"
        path.write_text(source)
        output = directory / ("claim.nvm" if backend == "bytecode" else "claim")
        args = [str(COMPILERS[backend]), str(path), "-o", str(output)]
        if backend == "bytecode":
            args.append("--emit-nvm")
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

    def test_shadow_execution_gap(self):
        source = "fn f() -> int { return 42 }\nshadow f { assert false }\nfn main() -> int { return 0 }\nshadow main { assert true }\n"
        for backend in COMPILERS:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory(prefix="nano-claims-") as tmp:
                compiled, output = self.compile_source(backend, source, Path(tmp))
                if backend == "c-seed":
                    self.assertGreater(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertIn(b"shadow", (compiled.stdout + compiled.stderr).lower())
                    self.assertFalse(output.exists())
                else:
                    # I record a known gap, not successful shadow enforcement.
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    self.assertEqual(self.execute(backend, output).returncode, 0)

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


if __name__ == "__main__":
    unittest.main()
