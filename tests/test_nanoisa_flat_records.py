"""I compare and execute the flat-record emitter subset across VM and AOT."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlatRecordEmitter(unittest.TestCase):
    def run_checked(self, *args):
        result = subprocess.run([str(a) for a in args], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_boolean_record_returns_match_and_execute(self):
        fixture = ROOT / "tests/nanoisa/fixtures/flat_record_results.nano"
        with tempfile.TemporaryDirectory(prefix="nano-flat-record-") as tmp:
            work = Path(tmp)
            seed, assembly, emitted = (work / n for n in ("seed.nvm", "emitter.nasm", "emitter.nvm"))
            self.run_checked(ROOT / "bin/nano_virt", fixture, "--emit-nvm", "--strip-debug", "-o", seed)
            self.run_checked(ROOT / "bin/nanoisa_emit", fixture, "-o", assembly)
            result = self.run_checked(ROOT / "tests/nanoisa/test_nanoisa_src_nano", seed, assembly,
                                     "make_options", "relay_options", "check_options", "main")
            self.assertIn("10 passed, 0 failed", result.stdout)
            self.run_checked(ROOT / "bin/nanoisa", "asm", assembly, "-o", emitted)
            for module in (seed, emitted):
                with self.subTest(module=module.name):
                    self.run_checked(ROOT / "bin/nano_vm", "--verify-only", module)
                    self.run_checked(ROOT / "bin/nano_vm", module)
                    source, binary = module.with_suffix(".c"), module.with_suffix(".exe")
                    self.run_checked(ROOT / "bin/nvm2c", module, "-o", source)
                    self.run_checked("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", binary)
                    self.run_checked(binary)

    def test_nested_record_results_remain_refused(self):
        with tempfile.TemporaryDirectory(prefix="nano-nested-record-") as tmp:
            work = Path(tmp)
            source, output = work / "nested.nano", work / "nested.nasm"
            source.write_text('struct Flags { enabled: bool }\nstruct Nested { child: Flags }\n'
                              'fn make() -> Nested { return Nested { child: Flags { enabled: true } } }\n'
                              'shadow make { let value: Nested = (make) assert value.child.enabled }\n'
                              'fn main() -> int { let value: Nested = (make) assert value.child.enabled return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            self.run_checked(ROOT / "bin/nano_virt", source, "--emit-nvm", "-o", work / "nested.nvm")
            self.run_checked(ROOT / "bin/nano_vm", work / "nested.nvm")
            result = subprocess.run([ROOT / "bin/nanoisa_emit", source, "-o", output],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside the pinned subset", result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
