"""I require native compilation to check root shadows before publication."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "bin/nanoc_stage2"


class NativeShadows(unittest.TestCase):
    def invoke(self, directory, source, *options, extra_env=None):
        path = directory / "input.nano"
        output = directory / "program"
        path.write_text(source)
        args = [str(COMPILER), str(path), "-o", str(output), *options]
        environment = dict(os.environ, TMPDIR=str(directory))
        environment.update(extra_env or {})
        process = subprocess.Popen(args, cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr), output

    def test_failed_shadow_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
            directory = Path(tmp)
            (directory / "program").write_bytes(b"preserve me")
            result, output = self.invoke(directory, "fn main() -> int { return 0 } shadow main { assert false }")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"shadow", result.stdout + result.stderr)
            self.assertEqual(output.read_bytes(), b"preserve me")
            self.assertFalse(list(directory.glob("nano_native_*")))

    def test_main_and_inferred_shadow_locals(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
            result, output = self.invoke(Path(tmp), 'fn main() -> int { (println "product") return 7 } shadow main { let expected = 7 assert (== (main) expected) }')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout, b"")
            self.assertIn(b"product\n", result.stderr)
            executed = subprocess.run([str(output)], capture_output=True, timeout=5)
            self.assertEqual(executed.returncode, 7)
            self.assertEqual(executed.stdout, b"product\n")

    def test_root_selection_and_imported_helper(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
            directory = Path(tmp)
            module = directory / "helper.nano"
            module.write_text("pub fn twice(x: int) -> int { return (* x 2) }\nshadow twice { assert false }\n")
            for expected in (6, 7):
                with self.subTest(expected=expected):
                    result, output = self.invoke(directory, f'module "{module}" as helper\nfn main() -> int {{ return 0 }}\nshadow main {{ assert (== (helper.twice 3) {expected}) }}')
                    self.assertEqual(result.returncode == 0, expected == 6, result.stdout + result.stderr)

    def test_shadow_only_source(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
            result, output = self.invoke(Path(tmp), "fn f() -> int { return 7 } shadow f { assert (== (f) 7) }")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(subprocess.run([str(output)], timeout=5).returncode, 0)

    def test_type_errors_before_source_publication(self):
        cases = [
            "shadow f { assert 7 }",
            'shadow f { let x: int = "wrong" assert true }',
            "shadow missing { assert true }",
            "shadow f { unsafe { assert 7 } }",
            "shadow f { let hidden: int = 7 assert true } shadow f { assert (== hidden 7) }",
            "shadow f { if true { let hidden: int = 7 } assert (== hidden 7) }",
            "shadow f { assert (== parameter 7) }",
        ]
        for shadow in cases:
            with self.subTest(shadow=shadow), tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
                directory = Path(tmp)
                (directory / "program").write_bytes(b"preserve source")
                result, output = self.invoke(directory, "fn f(parameter: int) -> int { return parameter } " + shadow,
                                             "--target", "c", extra_env={"NANO_CC": "/bin/false"})
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_bytes(), b"preserve source")

    def test_local_scopes_do_not_leak(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
            result, _ = self.invoke(Path(tmp), 'fn f() -> int { return 7 } shadow f { let x: int = 7 assert (== (f) x) } shadow f { let x: string = "seven" assert (== (str_length x) 5) }')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_source_only_requires_no_native_compiler(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
            result, output = self.invoke(Path(tmp), "fn main() -> int { return 0 } shadow main { assert false }",
                                         "--target", "c", extra_env={"NANO_CC": "/bin/false"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(b"without executing shadows", result.stdout)
            self.assertNotIn(b"__nano_shadow_", output.read_bytes())

    def test_ndebug_does_not_skip_shadows(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
            result, output = self.invoke(Path(tmp), "fn f() -> int { assert false return 0 } shadow f { (f) }",
                                         extra_env={"NANO_CFLAGS": "-DNDEBUG"})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_deadline_survives_foreign_alarm_cancellation(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
            directory = Path(tmp)
            helper = directory / "cancel.c"
            helper.write_text("#include <unistd.h>\nvoid cancel_shadow_alarm(void) { alarm(0); }\n")
            result, output = self.invoke(directory, "extern fn cancel_shadow_alarm() -> void\nfn f() -> int { return 0 } shadow f { unsafe { (cancel_shadow_alarm) } while true {} }",
                                         extra_env={"NANO_CFLAGS": str(helper)})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"after 10 seconds", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
