"""I compile my snapshot scanner without mistaking its literals for a PCH."""
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "obj/test_module_generation_probe"


class ModuleBuilderSelfCapture(unittest.TestCase):
    def build(self, module):
        env = os.environ.copy()
        env.pop("NANO_BUILD_CACHE", None)
        if sys.platform == "linux":
            env["NANO_AS_CAPTURE_HELPER"] = str(ROOT / "bin/nano_as_capture.so")
        return subprocess.run([PROBE, "build", module], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=120)

    def test_builder_source_retains_and_builds_its_own_marker(self):
        with tempfile.TemporaryDirectory(prefix="nano-self-capture-") as tmp:
            module = Path(tmp)
            (module / "module.json").write_text(json.dumps({
                "name": "self_capture",
                "c_sources": [str(ROOT / "src" / name) for name in
                              ("module_builder.c", "cJSON.c", "utf8.c", "runtime/module_build_dir.c")],
                "include_dirs": [str(ROOT / "src")],
                "cflags": ["-D_GNU_SOURCE"], "pkg_config": ["openssl"],
            }))
            result = self.build(module)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            artifact = Path(result.stdout.strip())
            self.assertTrue(artifact.is_file())
            # Adjacent C literals retain exactly the scanner's original runtime bytes.
            self.assertIn(b"#pragma GCC pch_preprocess", artifact.read_bytes())
            self.assertTrue(list(artifact.parent.glob("__snapshot_*.i")) or
                            list(artifact.parent.glob("__snapshot_*.s")))

    def test_gcc_still_refuses_a_literal_pch_marker(self):
        version = subprocess.run([shutil.which("cc"), "--version"], capture_output=True, check=True)
        if b"clang" in version.stdout.lower():
            self.skipTest("I exercise the GCC retained-preprocessor scanner")
        with tempfile.TemporaryDirectory(prefix="nano-literal-pch-") as tmp:
            module = Path(tmp)
            (module / "input.c").write_text('const char *marker = "#pragma GCC pch_preprocess";\n')
            (module / "module.json").write_text(json.dumps({"name": "literal", "c_sources": ["input.c"]}))
            result = self.build(module)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("I could not retain compiler and assembler inputs", result.stderr)
            self.assertFalse((module / ".build/current").exists())

    def test_gcc_still_captures_a_canonical_external_pch(self):
        compiler = shutil.which("cc")
        version = subprocess.run([compiler, "--version"], capture_output=True, check=True)
        if b"clang" in version.stdout.lower():
            self.skipTest("I exercise GCC's external PCH directive")
        with tempfile.TemporaryDirectory(prefix="nano-canonical-pch-") as tmp:
            module = Path(tmp)
            header, source = module / "external.h", module / "input.c"
            header.write_text("typedef char measured_type[42];\n")
            pch = Path(str(header) + ".gch")
            flags = ["-fPIC", "-D_POSIX_C_SOURCE=200809L"]
            built = subprocess.run([compiler, *flags, "-x", "c-header", header, "-o", pch],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            source.write_text("long long answer(void) { return sizeof(measured_type); }\n")
            captured = subprocess.run([compiler, *flags, "-E", "-fpch-preprocess", "-include", header, source],
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(captured.returncode, 0, captured.stderr)
            self.assertIn("#pragma GCC pch_preprocess", captured.stdout)
            snapshot = module / "captured.i"
            snapshot.write_text(captured.stdout)
            result = subprocess.run([PROBE, "capture-pch", snapshot, module], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("__pch_0_0_0.gch", snapshot.read_text())
            self.assertEqual((module / "__pch_0_0_0.gch").read_bytes(), pch.read_bytes())

    def test_gcc_still_refuses_an_unrepresentable_external_pch_path(self):
        compiler = shutil.which("cc")
        version = subprocess.run([compiler, "--version"], capture_output=True, check=True)
        if b"clang" in version.stdout.lower():
            self.skipTest("I exercise GCC's external PCH directive")
        with tempfile.TemporaryDirectory(prefix="nano-external-pch-") as tmp:
            module = Path(tmp)
            header = module / 'external"header.h'
            header.write_text("typedef char measured_type[42];\n")
            pch = Path(str(header) + ".gch")
            flags = ["-fPIC", "-D_POSIX_C_SOURCE=200809L"]
            built = subprocess.run([compiler, *flags, "-x", "c-header", header, "-o", pch],
                                   capture_output=True, text=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr)
            source = module / "input.c"
            source.write_text("long long answer(void) { return sizeof(measured_type); }\n")
            captured = subprocess.run([compiler, *flags, "-E", "-fpch-preprocess", "-include", header, source],
                                      capture_output=True, text=True, timeout=30)
            self.assertEqual(captured.returncode, 0, captured.stderr)
            self.assertIn("#pragma GCC pch_preprocess", captured.stdout)
            snapshot = module / "captured.i"
            snapshot.write_text(captured.stdout)
            result = subprocess.run([PROBE, "capture-pch", snapshot, module], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(snapshot.read_text(), captured.stdout)
            self.assertFalse(Path(str(snapshot) + ".pch").exists())


if __name__ == "__main__":
    unittest.main()
