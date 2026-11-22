"""I retain the types and headers needed by imported C declarations."""
from pathlib import Path
import os
import json
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ImportedCallbackTypedefs(unittest.TestCase):
    def own_header_fixture(self, work, headers):
        module = work / "bridge"
        module.mkdir()
        (module / "SDL.h").write_text("#ifndef NANO_OWN_HEADER_SDL_H\n"
            "#define NANO_OWN_HEADER_SDL_H\ntypedef unsigned int Uint32;\n"
            "Uint32 SDL_GetTicks(void);\n#endif\n")
        (module / "anchor.c").write_text('#include "SDL.h"\nUint32 SDL_GetTicks(void) { return 42; }\n')
        (module / "bridge.nano").write_text('module bridge\nextern fn SDL_GetTicks() -> int\n'
            'pub fn answer()->int{return 42}\nshadow answer{assert (== (answer) 42)}\n')
        (module / "module.json").write_text(json.dumps({'name': 'bridge', 'headers': headers,
            'c_sources': ['anchor.c'], 'include_dirs': ['.']}))
        source = work / "main.nano"
        source.write_text('module "bridge/bridge.nano" as bridge\n'
            'fn main()->int{assert (== (bridge.answer) 42) return 0}\n'
            'shadow main{assert (== (main) 0)}\n')
        return source

    def test_native_module_implementation_retains_its_own_header(self):
        with tempfile.TemporaryDirectory(prefix="nano-own-module-header-") as directory:
            work = Path(directory)
            source = self.own_header_fixture(work, ['SDL.h'])
            output = work / "program"
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source), '-o', str(output)],
                cwd=work, env=dict(os.environ, NANO_BUILD_CACHE=str(work / 'cache')),
                capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run = subprocess.run([str(output)], cwd=work, capture_output=True, timeout=10)
            self.assertEqual((run.returncode, run.stdout, run.stderr), (0, b'', b''))

    def test_undeclared_native_types_preserve_prior_output_on_refusal(self):
        with tempfile.TemporaryDirectory(prefix="nano-missing-module-header-") as directory:
            work = Path(directory)
            source = self.own_header_fixture(work, [])
            output = work / "program"
            output.write_bytes(b'prior-output')
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source), '-o', str(output)],
                cwd=work, env=dict(os.environ, NANO_BUILD_CACHE=str(work / 'cache')),
                capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(b'Uint32', result.stdout + result.stderr)
            self.assertEqual(output.read_bytes(), b'prior-output')

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
