"""I require performed effects to reach handlers, not merely parse."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''
effect Recorder { emit : int -> void }
let mut recorded: int = 0

fn send(value: int) -> void {
    perform Recorder.emit(value)
}

fn exercise() -> int {
    set recorded 0
    let ignored = handle { (send 7) } with {
        emit value -> { set recorded value }
    }
    return recorded
}

shadow send { assert (== (exercise) 7) }
shadow exercise { assert (== (exercise) 7) }
fn main() -> int {
    assert (== (exercise) 7)
    (println "I dispatched the effect.")
    return 0
}
shadow main { assert (== (main) 0) }
'''


class EffectExecution(unittest.TestCase):
    def test_handler_observes_perform(self):
        self.check_program(SOURCE)

    def test_ordered_multiple_and_zero_arguments(self):
        self.check_program('''
effect Recorder { pair : int int -> void, tick : void -> void }
let mut trace: int = 0
let mut recorded: int = 0
fn argument(value: int) -> int { set trace (+ (* trace 10) value) return value }
fn exercise() -> int {
    set trace 0 set recorded 0
    let first: int = 9
    let ignored = handle { perform Recorder.pair((argument 1) (+ first (argument 2))) } with {
        pair first second -> { set recorded (+ (* first 100) second) }
    }
    let ticked = handle { perform Recorder.tick() } with {
        tick -> { set recorded (+ recorded 1000) }
    }
    return (+ (* trace 10000) recorded)
}
shadow argument { assert (== (exercise) 121111) }
shadow exercise { assert (== (exercise) 121111) }
fn main() -> int {
    assert (== (exercise) 121111)
    (println "I dispatched the effect.")
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def check_program(self, source_text):
        for compiler, vm in (("nanoc_c", False), ("nano_virt", True)):
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix="nano-effect-execution-") as tmp:
                directory = Path(tmp)
                source = directory / "effect.nano"
                source.write_text(source_text)
                output = directory / ("effect.nvm" if vm else "effect")
                command = [str(ROOT / "bin" / compiler), str(source), "-o", str(output)]
                if vm: command.append("--emit-nvm")
                built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
                self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace"))
                command = [str(ROOT / "bin/nano_vm"), str(output)] if vm else [str(output)]
                run = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=10)
                self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))
                self.assertEqual(run.stdout, b"I dispatched the effect.\n")


if __name__ == "__main__":
    unittest.main()
