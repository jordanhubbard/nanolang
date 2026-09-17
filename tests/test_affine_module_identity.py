"""I keep same-spelled module types distinct through checking and native execution."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER_ROOT = Path(os.environ.get("NANOLANG_AFFINE_COMPILER_ROOT", ROOT / "bin"))
OWNERSHIP = r"(?i)(ownership|resource.{0,80}(scope|leak|live|consum)|moved value|after.{0,30}(mov|consum))"


class AffineModuleIdentity(unittest.TestCase):
    def check_modules(self, resource_body, accepted, reverse=False, nested=False, long_names=False):
        for compiler in os.environ.get("NANOLANG_AFFINE_COMPILERS", "nanoc_c,nanoc_stage1,nanoc_stage2").split(","):
            with self.subTest(compiler=compiler, reverse=reverse, nested=nested), tempfile.TemporaryDirectory(prefix="nano-affine-modules-") as directory:
                work = Path(directory)
                plain = "struct Handle { plain_value: int }\n"
                owned = "resource struct Handle { fd: int }\n"
                kind = "Handle"
                value = "Handle { fd: 7 }"
                plain_value = "Handle { plain_value: 7 }"
                plain_body = "let copied: Handle = value return (+ copied.plain_value value.plain_value)"
                if nested:
                    plain += "struct Envelope { inner: Handle }\n"
                    owned += "struct Envelope { inner: Handle }\n"
                    kind = "Envelope"
                    value = "Envelope { inner: Handle { fd: 7 } }"
                    plain_value = "Envelope { inner: Handle { plain_value: 7 } }"
                    plain_body = "let copied: Envelope = value return (+ copied.inner.plain_value value.inner.plain_value)"
                plain += f"pub fn inspect(value: {kind}) -> int {{ {plain_body} }}\n"
                plain += f"pub fn run_plain() -> int {{ return (inspect {plain_value}) }}\n"
                plain += f"shadow inspect {{ assert (== (inspect {plain_value}) 14) }}\nshadow run_plain {{ assert (== (run_plain) 14) }}\n"
                owned += f"fn close_owned(value: {kind}) -> int {{ "
                if nested:
                    owned += "let Envelope { inner } = value let Handle { fd } = inner return fd }\n"
                else:
                    owned += "let Handle { fd } = value return fd }\n"
                owned += f"shadow close_owned {{ assert (== (close_owned {value}) 7) }}\n"
                owned += f"pub fn run_owned() -> int {{ let value: {kind} = {value} {resource_body} }}\n"
                if accepted:
                    owned += "shadow run_owned { assert (== (run_owned) 7) }\n"
                plain_name = ("p" * 244 + "a.nano") if long_names else "plain.nano"
                owned_name = ("p" * 244 + "b.nano") if long_names else "owned.nano"
                (work / plain_name).write_text(plain)
                (work / owned_name).write_text(owned)
                imports = [f'module "{plain_name}" as ordinary', f'module "{owned_name}" as owning']
                if reverse:
                    imports.reverse()
                source = work / "main.nano"
                source.write_text("\n".join(imports) + "\nfn main() -> int { assert (== (ordinary.run_plain) 14) assert (== (owning.run_owned) 7) return 0 }\nshadow main { assert (== (main) 0) }\n")
                output = work / "program"
                output.write_bytes(b"prior artifact")
                result = subprocess.run([str(COMPILER_ROOT / compiler), str(source), "-o", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=120)
                messages = result.stdout + result.stderr
                if accepted:
                    self.assertEqual(result.returncode, 0, messages)
                    run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                else:
                    self.assertGreater(result.returncode, 0, messages)
                    self.assertRegex(messages, OWNERSHIP)
                    self.assertIn("owned.nano", messages)
                    self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_qualified_public_record_identity(self):
        fixtures = {
            'owned.nano': """resource struct Handle { fd: int }
pub struct Envelope { inner: Handle }
pub fn make_owner() -> Envelope { return Envelope { inner: Handle { fd: 7 } } }
pub fn close_owner(value: Envelope) -> int { let Envelope { inner } = value let Handle { fd } = inner return fd }
shadow make_owner { assert (== (close_owner (make_owner)) 7) }
shadow close_owner { assert (== (close_owner (make_owner)) 7) }
""",
            'plain.nano': """pub struct Envelope { value: int }
pub fn make_plain() -> Envelope { return Envelope { value: 5 } }
shadow make_plain { let value: Envelope = (make_plain) assert (== value.value 5) }
""",
            'main.nano': """module "owned.nano" as owning
module "owned.nano" as same_owner
module "plain.nano" as ordinary
fn route(value: owning.Envelope) -> same_owner.Envelope { return value }
shadow route { assert (== (owning.close_owner (route (owning.make_owner))) 7) }
fn main() -> int {
    let plain: ordinary.Envelope = (ordinary.make_plain)
    let copy: ordinary.Envelope = plain
    assert (== copy.value 5)
    return (- (owning.close_owner (route (owning.make_owner))) 7)
}
shadow main { assert (== (main) 0) }
""",
        }
        for compiler in os.environ.get("NANOLANG_AFFINE_COMPILERS", "nanoc_c,nanoc_stage1,nanoc_stage2").split(","):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-affine-qualified-") as directory:
                work = Path(directory)
                for name, source in fixtures.items():
                    (work / name).write_text(source)
                output = work / "program"
                result = subprocess.run([str(COMPILER_ROOT / compiler), str(work / "main.nano"), "-o", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_plain_copy_and_owned_move(self):
        for reverse in (False, True):
            self.check_modules("let moved: Handle = value return (close_owned moved)", True, reverse)

    def test_long_module_identity(self):
        for reverse in (False, True):
            self.check_modules("let moved: Handle = value return (close_owned moved)", True, reverse, long_names=True)

    def test_generic_annotation_metadata(self):
        for compiler in os.environ.get("NANOLANG_AFFINE_COMPILERS", "nanoc_c,nanoc_stage1,nanoc_stage2").split(","):
            for argument in ("int", "array<int>", "Handle"):
                with self.subTest(compiler=compiler, argument=argument), tempfile.TemporaryDirectory(prefix="nano-affine-generic-") as directory:
                    work = Path(directory)
                    source = work / "main.nano"
                    output = work / "program"
                    output.write_bytes(b"prior artifact")
                    if argument == "Handle":
                        program = """resource struct Handle { fd: int }
union Box<T> { Some { value: T }, None {} }
fn main() -> int { let boxed: Box<Handle> = Box.Some { value: Handle { fd: 7 } } return 0 }
shadow main { assert (== (main) 0) }
"""
                    else:
                        program = f"""union Box<T> {{ Some {{ value: int }}, None {{}} }}
fn read(value: Box<{argument}>) -> int {{
    match value {{ Some(v) => {{ return v.value }} None(n) => {{ return 0 }} }}
}}
shadow read {{ let boxed: Box<{argument}> = Box.Some {{ value: 7 }} assert (== (read boxed) 7) }}
fn main() -> int {{ let boxed: Box<{argument}> = Box.Some {{ value: 7 }} let copy: Box<{argument}> = boxed return (- (read copy) 7) }}
shadow main {{ assert (== (main) 0) }}
"""
                    source.write_text(program)
                    env = os.environ.copy()
                    env["MALLOC_PERTURB_"] = "165"
                    result = subprocess.run([str(COMPILER_ROOT / compiler), str(source), "-o", str(output)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
                    messages = result.stdout + result.stderr
                    if argument == "Handle":
                        self.assertGreater(result.returncode, 0, messages)
                        self.assertRegex(messages, OWNERSHIP)
                        self.assertEqual(output.read_bytes(), b"prior artifact")
                    else:
                        self.assertEqual(result.returncode, 0, messages)
                        run = subprocess.run([str(output)], capture_output=True, text=True, timeout=10)
                        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_foreign_record_collision_is_rejected(self):
        for compiler in os.environ.get("NANOLANG_AFFINE_COMPILERS", "nanoc_c,nanoc_stage1,nanoc_stage2").split(","):
            for reverse in (False, True):
                with self.subTest(compiler=compiler, reverse=reverse), tempfile.TemporaryDirectory(prefix="nano-affine-foreign-") as directory:
                    work = Path(directory)
                    (work / "foreign.nano").write_text("extern struct Handle { fd: int }\n")
                    (work / "local.nano").write_text("struct Handle { value: int }\n")
                    imports = ['module "foreign.nano" as foreign', 'module "local.nano" as ordinary']
                    if reverse:
                        imports.reverse()
                    source = work / "main.nano"
                    source.write_text("\n".join(imports) + "\nfn main() -> int { return 0 }\n")
                    output = work / "program"
                    output.write_bytes(b"prior artifact")
                    result = subprocess.run([str(COMPILER_ROOT / compiler), str(source), "-o", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=120)
                    self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("colliding foreign record declarations", result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_same_module_duplicate_is_rejected(self):
        for compiler in os.environ.get("NANOLANG_AFFINE_COMPILERS", "nanoc_c,nanoc_stage1,nanoc_stage2").split(","):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-affine-duplicate-") as directory:
                source = Path(directory) / "duplicate.nano"
                output = Path(directory) / "program"
                source.write_text("struct Handle { first: int }\nresource struct Handle { second: int }\nfn main() -> int { return 0 }\n")
                output.write_bytes(b"prior artifact")
                result = subprocess.run([str(COMPILER_ROOT / compiler), str(source), "-o", str(output)], cwd=ROOT, capture_output=True, text=True, timeout=120)
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertRegex(result.stdout + result.stderr, r"(?i)(already defined|twice in one module)")
                self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_nested_plain_copy_and_owned_move(self):
        for reverse in (False, True):
            self.check_modules("let moved: Envelope = value return (close_owned moved)", True, reverse, True)

    def test_unresolved_resource(self):
        for reverse in (False, True):
            self.check_modules("return 7", False, reverse)

    def test_resource_use_after_move(self):
        for reverse in (False, True):
            self.check_modules("let moved: Handle = value let closed: int = (close_owned moved) return value.fd", False, reverse)


if __name__ == "__main__":
    unittest.main()
