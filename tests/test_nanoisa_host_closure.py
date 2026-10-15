"""I exercise the Forth SEE manifest and its separate examples library rule."""
from pathlib import Path
import ctypes
import json
import sys
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NanoisaHostClosure(unittest.TestCase):
    def test_manifest_libraries_load_in_isolation(self):
        with tempfile.TemporaryDirectory(prefix="nano-host-closure-") as tmp:
            for module in ('nanoisa', 'forth_see'):
                with self.subTest(module=module):
                    directory = ROOT/'modules'/module
                    manifest = json.loads((directory/'module.json').read_text())
                    library = Path(tmp)/(module+'.so')
                    command = [os.environ.get('CC', 'cc'), '-std=c11', '-D_GNU_SOURCE',
                               '-fPIC', '-shared', '-I'+str(ROOT/'src'), '-I'+str(ROOT/'src/nanoisa'),
                               *[str(directory/source) for source in manifest['c_sources']],
                               '-lm', '-o', str(library)]
                    built = subprocess.run(command, capture_output=True, timeout=180)
                    self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
                    # A fresh process cannot mask omitted providers with earlier dlopen calls.
                    loaded = subprocess.run([sys.executable, '-c',
                        'import ctypes,sys; h=ctypes.CDLL(sys.argv[1]); '
                        'assert h.nl_nanoisa_assemble_text_save', str(library)],
                        capture_output=True, timeout=30)
                    self.assertEqual(loaded.returncode, 0, loaded.stdout+loaded.stderr)

    def test_forth_see_native_import(self):
        with tempfile.TemporaryDirectory(prefix="nano-forth-closure-") as tmp:
            source = Path(tmp) / "probe.nano"
            binary = Path(tmp) / "probe"
            missing = Path(tmp) / "missing.nvm"
            source.write_text(f'''module "{ROOT}/modules/forth_see/forth_see.nano"
fn main() -> int {{
    unsafe {{
        let detail: string = (nl_forth_see "dup" "{missing}")
        assert (str_contains detail "SEE: cannot load")
    }}
    return 0
}}
''')
            env = os.environ.copy()
            env["NANO_BUILD_CACHE"] = str(Path(tmp) / "cache")
            built = subprocess.run([ROOT / "bin/nanoc_c", source, "-o", binary],
                                   cwd=ROOT, env=env, capture_output=True, timeout=180)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            ran = subprocess.run([binary], cwd=ROOT, capture_output=True, timeout=30)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)

    def test_examples_shared_library_loads(self):
        # I exercise the examples rule as well as the compiler's module manifest.
        library = ROOT / "modules/forth_see/.build/libforth_see.so"
        built = subprocess.run([os.environ.get("MAKE", "make"), "-C", ROOT / "examples",
                                "../modules/forth_see/.build/libforth_see.so"],
                               cwd=ROOT, capture_output=True, timeout=180)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        host = ctypes.CDLL(str(library))
        see = host.nl_forth_see
        see.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        see.restype = ctypes.c_char_p
        with tempfile.TemporaryDirectory(prefix="nano-forth-load-") as tmp:
            detail = see(b"dup", os.fsencode(Path(tmp) / "missing.nvm"))
            self.assertIn(b"SEE: cannot load", detail)


if __name__ == "__main__":
    unittest.main()
