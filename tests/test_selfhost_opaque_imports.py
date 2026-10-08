"""I retain opaque declarations through the canonical import-merging driver."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class OpaqueImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="nano-opaque-imports-"))
        print("I retain opaque import controls at", cls.work, flush=True)
        override = os.environ.get("NANOLANG_OPAQUE_COMPILER")
        cls.compilers = [Path(override)] if override else [ROOT / "bin/nanoc_stage1", ROOT / "bin/nanoc_stage2"]
        cls.sequence = 0

    @classmethod
    def command(cls, *args, success=True):
        cls.sequence += 1
        result = subprocess.run(list(map(str,args)), cwd=ROOT, capture_output=True, text=True, timeout=180)
        (cls.work / f"command-{cls.sequence}.log").write_text(repr(list(map(str,args))) + "\n" + result.stdout + result.stderr)
        if (result.returncode == 0) != success:
            raise AssertionError(f"{args}: {result.returncode}\n{result.stdout}\n{result.stderr}")
        return result

    def test_root_and_imported_declarations(self):
        module = self.work / "handles.nano"
        module.write_text("opaque type Handle\npub fn ready()->int { return 0 }\nshadow ready { assert (== (ready) 0) }\n")
        for imported in (False, True):
            prefix = f'from "{module}" import ready\n' if imported else "opaque type Handle\n"
            for compiler in self.compilers:
                with self.subTest(imported=imported, compiler=str(compiler)):
                    source = self.work / "main.nano"
                    output = self.work / "main.nvm"
                    good = prefix + "fn main()->int { let handle:Handle = 0 return 0 } shadow main { assert true }\n"
                    source.write_text(good)
                    self.command(compiler, source, "--emit-nvm", "-o", output)
                    self.command(ROOT / "bin/nano_vm", "--verify-only", output)
                    self.command(ROOT / "bin/nano_vm", output)
                    original = output.read_bytes()
                    source.write_text(good.replace("handle:Handle = 0", "handle:Handle = 1"))
                    self.command(compiler, source, "--emit-nvm", "-o", output, success=False)
                    self.assertEqual(output.read_bytes(), original)
                    source.write_text(good)
                    self.command(compiler, source, "--emit-nvm", "-o", output)
                    self.command(ROOT / "bin/nano_vm", output)

    def test_unknown_opaque_name_refused(self):
        source = self.work / "unknown.nano"
        source.write_text("fn main()->int { let handle:Missing = 0 return 0 } shadow main { assert true }\n")
        for compiler in self.compilers:
            with self.subTest(compiler=str(compiler)):
                output = self.work / "unknown.nvm"
                output.write_bytes(b"prior output")
                self.command(compiler, source, "--emit-nvm", "-o", output, success=False)
                self.assertEqual(output.read_bytes(), b"prior output")
