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
    def check_modules(self, resource_body, accepted, reverse=False, nested=False):
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
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
                (work / "plain.nano").write_text(plain)
                (work / "owned.nano").write_text(owned)
                imports = ['module "plain.nano" as ordinary', 'module "owned.nano" as owning']
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
                    self.assertEqual(output.read_bytes(), b"prior artifact")

    def test_plain_copy_and_owned_move(self):
        for reverse in (False, True):
            self.check_modules("let moved: Handle = value return (close_owned moved)", True, reverse)

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
