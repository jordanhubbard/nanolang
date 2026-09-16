"""I execute dependency shadows through the retained foreign callback bridge."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CallbackShadows(unittest.TestCase):
    def compile_fixture(self, succeeds, execution):
        with tempfile.TemporaryDirectory(prefix="nano-callback-shadows-") as directory:
            work = Path(directory)
            (work / "module.json").write_text(json.dumps({
                "name": "callback_shadow_fixture", "version": "1.0.0",
                "c_sources": ["fixture.c"],
                "callback_adapters": {"invoke": {"symbol": "invoke_retained",
                    "abi": "retained_v1", "execution": execution}}}))
            header = ROOT / "src/runtime/nano_callback.h"
            (work / "fixture.c").write_text(f'#include "{header}"\n' + """
int64_t invoke(int64_t (*fn)(int64_t), int64_t value) { return fn(value); }
int64_t invoke_retained(NanoCallbackV1 *fn, int64_t value) {
    NanoCallbackValue arg = {.tag=NANO_CALLBACK_INT, .as.integer=value}, result;
    if (fn->invoke(fn, &arg, 1, &result) != NANO_CALLBACK_OK) return -1;
    return result.as.integer;
}
""")
            expected = 42 if succeeds else 99
            (work / "fixture.nano").write_text("""extern fn invoke(callback: fn(int) -> int, value: int) -> int
fn increment(value: int) -> int { return (+ value 1) }
shadow increment { assert (== (increment 41) 42) }
fn callback_check() -> int { unsafe { return (invoke increment 41) } }
""" + f'shadow callback_check {{ assert (== (callback_check) {expected}) }}\n')
            source = work / "main.nano"
            source.write_text('module "fixture.nano" as Fixture\n'
                              'fn main() -> int { return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            output = work / "program"
            report = work / "shadows.json"
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source),
                                     "-o", str(output), "--llm-shadow-json", str(report)], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertTrue(report.exists(), result.stdout + result.stderr)
            status = json.loads(report.read_text())
            self.assertEqual(status["success"], succeeds)
            self.assertTrue(status["completed"])
            self.assertEqual(status["backend"], "nano_vm")
            if succeeds:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                run = subprocess.run([str(output)], capture_output=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr)
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("I failed a shadow", result.stderr)
                self.assertFalse(output.exists())

    def test_owner_callback_dependency(self):
        self.compile_fixture(True, "owner")

    def test_worker_callback_dependency(self):
        self.compile_fixture(True, "worker")

    def test_failed_dependency_prevents_publication(self):
        self.compile_fixture(False, "worker")


if __name__ == "__main__":
    unittest.main()
