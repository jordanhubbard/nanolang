"""I preserve effects when a local binds the sole void value."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''
let mut trace: int = 0
fn record(value: int) -> void { set trace (+ (* trace 10) value) }
fn exercise() -> void {
    let first = (record 1)
    let mut second: void = (record 2)
    set second (record 3)
    set second first
    let third: void = second
    return third
}
fn check() -> int {
    set trace 0
    let ignored = (exercise)
    return trace
}
shadow record { assert (== (check) 123) }
shadow exercise { assert (== (check) 123) }
shadow check { assert (== (check) 123) }
fn main() -> int { assert (== (check) 123) return 0 }
shadow main { assert (== (main) 0) }
'''


class VoidBindings(unittest.TestCase):
    def test_effects_and_bound_values(self):
        for compiler, vm in (("nanoc_c", False), ("nano_virt", True)):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-void-bindings-") as tmp:
                source = Path(tmp) / "void.nano"
                source.write_text(SOURCE)
                artifact = Path(tmp) / ("program.nvm" if vm else "program")
                command = [str(ROOT / "bin" / compiler), str(source), "-o", str(artifact)]
                if vm:
                    command.append("--emit-nvm")
                built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
                self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace"))
                command = [str(ROOT / "bin/nano_vm"), str(artifact)] if vm else [str(artifact)]
                run = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))
                self.assertEqual(run.stdout, b"")


if __name__ == "__main__":
    unittest.main()
