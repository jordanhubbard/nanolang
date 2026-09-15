"""I check generated native foreign references before execution or capture."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeArrayAbi(unittest.TestCase):
    def run_ok(self, command):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        if result.returncode and "--keep-c" in command:
            result.stderr += result.stdout
            output = Path(command[-1]).with_suffix(".c")
            if output.exists():
                result.stderr += "\nABI declarations: " + "\n".join(
                    line for line in output.read_text().splitlines() if "nano_array_abi" in line)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_generated_references(self):
        compiler = os.environ.get("NANO_TEST_NATIVE_COMPILER", str(ROOT / "bin/nanoc_c"))
        cc = shlex.split(os.environ.get("CC", "cc"))
        with tempfile.TemporaryDirectory(prefix="nano-native-array-abi-") as tmp:
            directory = Path(tmp)
            foreign = directory / "foreign.c"
            foreign.write_text('''
                #include "runtime/dyn_array.h"
                #include <stdio.h>
                static DynArray empty = {.elem_type=ELEM_INT, .elem_size=8};
                #ifdef ARRAY_PARAMETER
                int64_t probe(DynArray *a) { puts("foreign entered"); return a->length == 1 ? 0 : 1; }
                #else
                DynArray *probe(void) { puts("foreign entered"); return &empty; }
                #endif
                #ifdef ABI_VERSION
                const uint32_t probe__nano_array_abi = ABI_VERSION;
                #endif
                ''')
            module = directory / "foreign.nano"
            module.write_text("pub extern fn probe() -> array<int>\n")
            for route in ("direct", "qualified", "value", "qualified_value", "parameter"):
                source = directory / (route + ".nano")
                declaration = (f'module "{module}" as foreign\n' if route.startswith("qualified")
                               else "extern fn probe() -> array<int>\n")
                if route == "parameter":
                    declaration = "extern fn probe(a: array<int>) -> int\n"
                body = ("let f: fn() -> array<int> = probe\nlet a: array<int> = (f)"
                        if route == "value" else
                        "let a: array<int> = (" + ("foreign." if route == "qualified" else "") + "probe)")
                if route == "qualified_value":
                    body = "let f: fn() -> array<int> = foreign.probe\nlet a: array<int> = (f)"
                result_body = body + "\nreturn (array_length a)"
                if route == "parameter":
                    result_body = "let a: array<int> = [7]\nreturn (probe a)"
                source.write_text(declaration + "\nfn main() -> int { unsafe {\n" + result_body + "\n} }\n")
                if route == "qualified_value":
                    rejected = directory / "qualified_value"
                    result = subprocess.run([compiler, str(source), "-o", str(rejected)],
                                            cwd=ROOT, capture_output=True, text=True, timeout=60)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Field access requires a struct value", result.stderr)
                    self.assertFalse(rejected.exists())
                    continue
                # I exercise deliberately incompatible FFI in subprocesses;
                # these are native boundary fixtures, not interpreter shadows.
                variants = [(version, False) for version in (None, 1, 99)]
                if route == "direct":
                    variants += [(version, True) for version in (None, 1, 99)]
                for version, static in variants:
                    with self.subTest(route=route, version=version, static=static):
                        library = directory / ("libfixture.dylib" if sys.platform == "darwin" else "libfixture.so")
                        library.unlink(missing_ok=True)
                        archive = directory / "libfixture.a"
                        archive.unlink(missing_ok=True)
                        shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
                        if static:
                            shared = ["-c"]
                            library = directory / "foreign.o"
                        self.run_ok([*cc, "-Isrc", *shared, str(foreign),
                                     *(["-DARRAY_PARAMETER"] if route == "parameter" else []),
                                     *([] if version is None else [f"-DABI_VERSION={version}"]),
                                     "-o", str(library)])
                        if static:
                            self.run_ok(["ar", "rcs", str(archive), str(library)])
                        executable = directory / "program"
                        self.run_ok([compiler, str(source), "--verbose", "--keep-c", "-L", tmp,
                                     "-lfixture", "-o", str(executable)])
                        run_env = dict(os.environ)
                        run_env["LD_LIBRARY_PATH"] = tmp + ":" + run_env.get("LD_LIBRARY_PATH", "")
                        result = subprocess.run([str(executable)], env=run_env, capture_output=True, text=True, timeout=15)
                        if version == 99:
                            self.assertNotEqual(result.returncode, 0)
                            self.assertNotIn("foreign entered", result.stdout)
                            self.assertIn("native array ABI", result.stderr)
                        else:
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertIn("foreign entered", result.stdout)


if __name__ == "__main__":
    unittest.main()
