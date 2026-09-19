"""I preserve exact scalar u8 spelling and conversions in my native C route."""

from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPILERS = Path(os.environ.get("NANOLANG_U8_COMPILER_ROOT", ROOT / "bin"))
FIXTURE = ROOT / "tests/nanoisa/fixtures/selfhost_native_u8.nano"


class SelfhostNativeU8(unittest.TestCase):
    def test_cseed_stage1_and_stage2_compile_and_run(self):
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(
                prefix="nano-native-u8-"
            ) as tmp:
                output = Path(tmp) / "program"
                result = subprocess.run(
                    [COMPILERS / compiler, FIXTURE, "-o", output],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run(
                    [output], capture_output=True, text=True, timeout=10
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failed_shadow_preserves_prior_output(self):
        source = FIXTURE.read_text().replace(
            "assert (cast_bool high_value)",
            "assert (not (cast_bool high_value))",
        )
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(
                prefix="nano-native-u8-refusal-"
            ) as tmp:
                path = Path(tmp) / "input.nano"
                output = Path(tmp) / "program"
                path.write_text(source)
                output.write_bytes(b"previous accepted output")
                result = subprocess.run(
                    [COMPILERS / compiler, path, "-o", output],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("shadow", (result.stdout + result.stderr).lower())
                self.assertEqual(output.read_bytes(), b"previous accepted output")

    def test_out_of_range_literal_preserves_prior_output(self):
        source = """fn main() -> int {
    let value: u8 = 256
    return (cast_int value)
}
shadow main { assert (== (main) 0) }
"""
        for compiler in ("nanoc_c", "nanoc_stage1", "nanoc_stage2"):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(
                prefix="nano-native-u8-range-"
            ) as tmp:
                path = Path(tmp) / "input.nano"
                output = Path(tmp) / "program"
                path.write_text(source)
                output.write_bytes(b"previous accepted output")
                result = subprocess.run(
                    [COMPILERS / compiler, path, "-o", output],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                diagnostic = (result.stdout + result.stderr).lower()
                self.assertTrue(
                    any(word in diagnostic for word in ("u8", "range", "shadow")),
                    diagnostic,
                )
                self.assertEqual(output.read_bytes(), b"previous accepted output")

    def test_runtime_cast_helpers_build_under_strict_sanitizers(self):
        source = r"""
#include <stdbool.h>
#include <stdint.h>

static int64_t nl_cast_int_from_int(int64_t x) { return x; }
static bool nl_cast_bool(int64_t x) { return x != 0; }
static bool nl_cast_bool_from_float(double x) { return x != 0.0; }

int main(void) {
    uint8_t low = 0;
    uint8_t high = UINT8_MAX;
    if (nl_cast_int_from_int(low) != 0) return 1;
    if (nl_cast_int_from_int(high) != UINT8_MAX) return 2;
    if (nl_cast_bool(low)) return 3;
    if (!nl_cast_bool(high)) return 4;
    if (!nl_cast_bool_from_float(0.5)) return 5;
    if (nl_cast_bool_from_float(0.0)) return 6;
    return 0;
}
"""
        configured = os.environ.get("NANOLANG_U8_C_COMPILERS", "cc")
        compilers = [item for item in configured.replace(",", " ").split() if item]
        native_flags = shlex.split(os.environ.get("NMS_NATIVE_CLANG_FLAGS", ""))
        leak_detection = os.environ.get(
            "NANOLANG_U8_SANITIZER_LEAKS", "0" if os.uname().sysname == "Darwin" else "1"
        )
        for compiler in compilers:
            with self.subTest(c_compiler=compiler), tempfile.TemporaryDirectory(
                prefix="nano-u8-c-"
            ) as tmp:
                c_path = Path(tmp) / "program.c"
                output = Path(tmp) / "program"
                c_path.write_text(source)
                result = subprocess.run(
                    [
                        compiler,
                        *native_flags,
                        "-std=c11",
                        "-O1",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "-fsanitize=address,undefined",
                        "-fno-sanitize-recover=all",
                        c_path,
                        "-o",
                        output,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                environment = dict(
                    os.environ,
                    ASAN_OPTIONS=f"detect_leaks={leak_detection}:halt_on_error=1",
                    UBSAN_OPTIONS="halt_on_error=1",
                )
                result = subprocess.run(
                    [output],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=environment,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
