"""I keep my Darwin SDK/libffi selection coherent and overridable."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MakeToolchain(unittest.TestCase):
    def flags(self, *assignments):
        makefile = """include Makefile.gnu
.PHONY: print-toolchain-flags
print-toolchain-flags:
\t@printf '%s\\n' "$(LIBFFI_CFLAGS)" "$(LIBFFI_LIBS)"
"""
        env = os.environ.copy()
        env.pop("LIBFFI_CFLAGS", None)
        env.pop("LIBFFI_LIBS", None)
        result = subprocess.run(
            ["make", "--no-print-directory", "-s", "-f", "-",
             "print-toolchain-flags", *assignments],
            cwd=ROOT, env=env, input=makefile, capture_output=True,
            text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.splitlines()

    def test_darwin_sdkroot_supplies_matching_system_libffi_headers(self):
        with tempfile.TemporaryDirectory(prefix="nano-darwin-sdk-") as tmp:
            sdk = Path(tmp) / "Selected.sdk"
            include = sdk / "usr/include/ffi"
            include.mkdir(parents=True)
            (include / "ffi.h").write_text("/* fixture */\n")
            cflags, _ = self.flags("UNAME_S=Darwin", f"SDKROOT={sdk}")
            self.assertEqual(cflags, f"-I'{include}'")

    def test_explicit_libffi_overrides_remain_exact(self):
        cflags, libraries = self.flags(
            "UNAME_S=Darwin",
            "LIBFFI_CFLAGS=-I/explicit/ffi -DEXPLICIT_FFI=1",
            "LIBFFI_LIBS=-L/explicit/lib -lffi_explicit",
        )
        self.assertEqual(cflags, "-I/explicit/ffi -DEXPLICIT_FFI=1")
        self.assertEqual(libraries, "-L/explicit/lib -lffi_explicit")

    def test_native_test_compiler_uses_declared_selection_and_cc_fallback(self):
        makefile = """include Makefile.gnu
.PHONY: print-native-test-cc
print-native-test-cc:
\t@printf '%s\\n' "$(NATIVE_TEST_CC)"
"""
        env = os.environ.copy()
        env.pop("NANO_NATIVE_TEST_CC", None)
        for assignments, expected in (
            (("CC=fallback-cc",), "fallback-cc"),
            (("CC=fallback-cc", "NANO_NATIVE_TEST_CC=selected-cc -isysroot /sdk"),
             "selected-cc -isysroot /sdk"),
        ):
            result = subprocess.run(
                ["make", "--no-print-directory", "-s", "-f", "-",
                 "print-native-test-cc", *assignments],
                cwd=ROOT, env=env, input=makefile, capture_output=True,
                text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), expected)

        recipe = (ROOT / "Makefile.gnu").read_text()
        target = recipe.split("test-parser-recovery:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("$(NATIVE_TEST_CC) $(CFLAGS)", target)
        self.assertNotIn("$(CC) $(CFLAGS)", target)

    @unittest.skipUnless(sys.platform == "darwin", "I select an Apple SDK only on Darwin")
    def test_default_darwin_headers_belong_to_the_active_sdk(self):
        cflags, _ = self.flags()
        sdk = os.environ.get("SDKROOT", "").strip()
        if not sdk:
            developer = os.environ.get("DEVELOPER_DIR")
            env = dict(os.environ, **({"DEVELOPER_DIR": developer} if developer else {}))
            sdk = subprocess.check_output(
                ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                env=env, text=True, timeout=10,
            ).strip()
        self.assertEqual(cflags, f"-I'{sdk}/usr/include/ffi'")


if __name__ == "__main__":
    unittest.main()
