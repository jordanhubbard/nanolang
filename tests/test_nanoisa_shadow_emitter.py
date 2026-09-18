"""I execute bound-Parser shadow modules; supervision remains a driver task."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get("NANOC", ROOT / "bin/nanoc_c")).resolve()


class ShadowEmitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nano-shadow-emitter-")
        cls.work = Path(cls.tmp.name)
        source = cls.work / "driver.nano"
        source.write_text((ROOT / "tests/nanoisa/fixtures/shadow_module_driver.nano.txt").read_text())
        cls.tool = cls.work / "driver"
        cls.command(COMPILER, source, "-o", cls.tool)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def command(*args, expected=0):
        result = subprocess.run([str(a) for a in args], cwd=ROOT,
                                text=True, capture_output=True, timeout=180)
        if result.returncode != expected:
            raise AssertionError(f"{args}: {result.returncode}\n{result.stdout}\n{result.stderr}")
        return result

    def emit(self, source, first=0, mode="raw", expected=0):
        path = self.work / "source.nano"
        path.write_text(source)
        return self.command(self.tool, path, first, mode, expected=expected)

    def execute(self, assembly, success=True):
        asm, module = self.work / "shadow.nasm", self.work / "shadow.nvm"
        asm.write_text(assembly)
        self.command(ROOT / "bin/nanoisa", "asm", asm, "-o", module)
        self.command(ROOT / "bin/nano_vm", "--verify-only", module)
        result = subprocess.run([ROOT / "bin/nano_vm", module], cwd=ROOT,
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def test_order_shared_globals_local_scopes_and_callable_main(self):
        source = '''let mut count: int = (initial)
fn initial() -> int { (println "init") return 4 }
fn main() -> int { return count }
fn unused() -> float { return 1.5 }
let values: array<int> = (array_new 3 7)
shadow main { let local: int = 7 assert (== (main) 4) assert (== (at values 2) 7) set count local }
shadow main { let local: int = 2 assert (== (main) 7) set count (+ count local) }
shadow main { assert (== count 9) (println "done") }
'''
        assembly = self.emit(source).stdout
        self.assertEqual(assembly, self.emit(source).stdout)
        self.assertNotIn(".function unused", assembly)
        self.assertEqual(self.execute(assembly).stdout, "init\ndone\n")

    def test_suffix_and_empty_selection_keep_initialization(self):
        source = '''let count: int = (initial)
fn initial() -> int { (println "init") return 4 }
fn work() -> int { return count }
shadow work { assert false }
shadow work { assert (== (work) 4) }
'''
        self.execute(self.emit(source).stdout, success=False)
        self.assertEqual(self.execute(self.emit(source, 1).stdout).stdout, "init\n")
        self.assertEqual(self.execute(self.emit(source, 2).stdout).stdout, "init\n")
        self.execute(self.emit("", 0).stdout)
        for first in (-1, 3):
            self.assertIn("first-shadow", self.emit(source, first, expected=1).stdout)

    def test_bound_duplicate_names_and_internal_calls(self):
        source = '''fn dep_a_value(n: int) -> int { if (== n 0) { return 37 } return (value (- n 1)) }
shadow value { assert (== (value 2) 37) }

fn dep_b_value(n: int) -> int { if (== n 0) { return 12 } return (value (- n 1)) }
shadow value { assert (== (value 2) 12) }

fn main() -> int { return 0 }
shadow main { assert (== (first.value 1) 37) assert (== (second.value 1) 12) }
'''
        assembly = self.emit(source, mode="bound").stdout
        self.assertIn("CALL dep_a_value", assembly)
        self.assertIn("CALL dep_b_value", assembly)
        self.execute(assembly)
        # The same process API's own shadows also exercise reset boundaries.
        self.execute(self.emit(source, 2, "bound").stdout)

    def test_synthetic_names_do_not_capture_user_functions(self):
        source = '''fn __nanoisa_shadow_entry() -> int { return 3 }
fn __nanoisa_shadow_0() -> int { return 7 }
fn __nanoisa_shadow_0_() -> int { return 11 }
shadow __nanoisa_shadow_0 { assert (== (__nanoisa_shadow_0) 7) assert (== (__nanoisa_shadow_0_) 11) assert (== (__nanoisa_shadow_entry) 3) }
'''
        assembly = self.emit(source).stdout
        self.assertIn(".function __nanoisa_shadow_0__ ", assembly)
        self.execute(assembly)

    def test_unused_float_function_still_runs_selected_shadow(self):
        source = '''fn unused() -> float { return 1.5 }
shadow unused { assert (> (unused) 1.0) }
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
'''
        assembly = self.emit(source).stdout
        self.assertIn(".function unused", assembly)
        self.assertIn("F64_GT", assembly)
        self.execute(assembly)
        failed = source.replace("assert (> (unused) 1.0)", "assert false")
        self.execute(self.emit(failed).stdout, success=False)

    def test_reachable_unsupported_shadow_refused(self):
        source = '''extern fn unsupported() -> array<array<float>>
fn unused() -> array<array<float>> { return (unsupported) }
shadow unused { let values: array<array<float>> = (unused) }
'''
        refusal = self.emit(source, expected=1)
        self.assertIn("I cannot lower shadow unused at merged line", refusal.stdout)
        self.execute(self.emit(source, 1).stdout)
