"""I require native compilation to check selected shadows before publication."""
import json
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
                    result, output = self.invoke(directory, f'module "{module}" as helper\nfn main() -> int {{ return 0 }}\nshadow main {{ assert (== (helper.twice 3) {expected}) }}', "--root-shadows-only")
                    self.assertEqual(result.returncode == 0, expected == 6, result.stdout + result.stderr)

    def test_dependencies_run_by_default(self):
        for transitive in (False, True):
            with self.subTest(transitive=transitive), tempfile.TemporaryDirectory(prefix="nano-native-imports-") as tmp:
                directory = Path(tmp)
                leaf = directory / "leaf.nano"
                leaf.write_text('pub fn answer() -> int { return 42 }\nshadow answer { assert false }\n')
                dependency = leaf
                if transitive:
                    dependency = directory / "middle.nano"
                    dependency.write_text(f'module "{leaf}" as lib\npub fn answer() -> int {{ return (lib.answer) }}\n')
                source = f'module "{dependency}" as lib\nfn main() -> int {{ return (lib.answer) }}\n'
                for options in ((), ("--test-imports",), ("--root-shadows-only",)):
                    (directory / "program").write_bytes(b"prior output")
                    result, output = self.invoke(directory, source, *options)
                    self.assertEqual(result.returncode == 0, options == ("--root-shadows-only",), result.stdout + result.stderr)
                    if result.returncode:
                        self.assertEqual(output.read_bytes(), b"prior output")

    def test_diamond_physical_identity_and_owner_order(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-diamond-") as tmp:
            directory = Path(tmp)
            physical = directory / "physical"
            physical.mkdir()
            (physical / "child.nano").write_text('pub fn value() -> int { return 9 }\nshadow value { (println "child") assert (== (value) 9) }\n')
            leaf = physical / "leaf.nano"
            leaf.write_text('module "./child.nano" as child\npub fn answer() -> int { return (child.value) }\nshadow answer { (println "leaf") assert (== (answer) 9) }\n')
            link = directory / 'leaf"link.nano'
            link.symlink_to(leaf)
            for name, path in (("left", str(link)), ("right", str(physical) + "/./leaf.nano")):
                (directory / f"{name}.nano").write_text(f'module {json.dumps(path)} as lib\npub fn answer() -> int {{ return (lib.answer) }}\nshadow answer {{ (println "{name}") assert (== (answer) 9) }}\n')
            source = f'module "{directory / "left.nano"}" as left\nmodule "{directory / "right.nano"}" as right\nfn main() -> int {{ return (+ (left.answer) (right.answer)) }}\nshadow main {{ (println "root") assert (== (main) 18) }}\n'
            result, output = self.invoke(directory, source)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stderr.splitlines(), [b"child", b"leaf", b"left", b"right", b"root"])
            run = subprocess.run([str(output)], capture_output=True, timeout=5)
            self.assertEqual(run.returncode, 18)
            self.assertEqual(run.stdout + run.stderr, b"")

    def test_dependency_types_and_source_only_selection(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-import-types-") as tmp:
            directory = Path(tmp)
            leaf = directory / "leaf.nano"
            source = f'module "{leaf}" as lib\nfn main() -> int {{ return (lib.answer) }}\n'
            for body in ('assert false', 'let bad: int = "wrong"'):
                leaf.write_text(f'pub fn answer() -> int {{ return 42 }}\nshadow answer {{ {body} }}\n')
                (directory / "program").write_bytes(b"prior source")
                result, output = self.invoke(directory, source, "--target", "c", extra_env={"NANO_CC": "/bin/false"})
                self.assertEqual(result.returncode == 0, body == 'assert false', result.stdout + result.stderr)
                if result.returncode:
                    self.assertEqual(output.read_bytes(), b"prior source")
                else:
                    self.assertNotIn(b"__nano_shadow_", output.read_bytes())

    def test_literal_braces_and_declared_record_arrays(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-literals-") as tmp:
            result, output = self.invoke(Path(tmp),
                'struct Point { x: int }\n'
                'fn make_point() -> Point { return Point { x: 9 } }\n'
                'shadow make_point { assert (== (make_point).x 9) }\n'
                'fn append_name(names: array<string>) -> int { let result = (array_push names "ok") return (array_length result) }\n'
                'shadow append_name { assert (== (append_name []) 1) assert (== (append_name [(str_concat "a" "b")]) 2) }\n'
                'fn main() -> int { return 0 }\n'
                'shadow main { let name: string = "expanded"\n'
                'assert (== "{name}" (str_concat "{" "name}"))\n'
                'assert (== f"{name}" "expanded")\n'
                'let direct: array<Point> = [Point { x: 7 }]\n'
                'let called: array<Point> = [(make_point)]\n'
                'let nested: array<string> = (array_push (array_push [] "first") "second")\n'
                'assert (== (at nested 0) "first") assert (== (at nested 1) "second")\n'
                'assert (== (at direct 0).x 7) assert (== (at called 0).x 9) }\n')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(subprocess.run([str(output)], timeout=5).returncode, 0)

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

    def test_early_zero_exit_preserves_output(self):
        with tempfile.TemporaryDirectory(prefix="nano-native-shadows-") as tmp:
            directory = Path(tmp)
            helper = directory / "early.c"
            helper.write_text("#include <stdlib.h>\nvoid leave_shadows_early(void) { exit(0); }\n")
            (directory / "program").write_bytes(b"prior output")
            result, output = self.invoke(directory,
                "extern fn leave_shadows_early() -> void\nfn f() -> int { return 0 }\n"
                "shadow f { unsafe { (leave_shadows_early) } assert false }",
                extra_env={"NANO_CFLAGS": str(helper)})
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(b"shadow", result.stderr)
            self.assertEqual(output.read_bytes(), b"prior output")


if __name__ == "__main__":
    unittest.main()
