"""I emit imported callback types before their C declarations."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ImportedCallbackTypedefs(unittest.TestCase):
    def test_implicit_public_and_opaque_callback_signatures(self):
        with tempfile.TemporaryDirectory(prefix="nano-callback-types-") as directory:
            work = Path(directory)
            module = work / "callbacks.nano"
            module.write_text("""opaque type Context
opaque type Buffer
extern fn install(callback: fn(Context, Buffer, int) -> void, context: Context) -> int
extern fn select_callback(callback: fn(int) -> bool) -> void
""")
            source = work / "main.nano"
            source.write_text('module "callbacks.nano" as Callbacks\n'
                              'fn main() -> int { return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            generated = work / "program.c"
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source),
                                     "--keep-c", "-o", str(work / "program")],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            text = generated.read_text()
            self.assertLess(text.index("typedef void (*FnType_"), text.index("extern int64_t install("))
            self.assertRegex(text, r"typedef void \(\*FnType_\d+\)\(void\*, void\*, int64_t\)")
            checked = subprocess.run([*shlex.split(os.environ.get("CC", "cc")),
                                      "-std=c11", "-Werror=incompatible-pointer-types",
                                      "-fsyntax-only", "-I", str(ROOT / "src"), str(generated)],
                                     cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)


if __name__ == "__main__":
    unittest.main()
