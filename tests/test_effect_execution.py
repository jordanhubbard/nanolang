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

    def test_lexical_return_and_final_expression(self):
        self.check_program('''
effect Ask { ask : int -> int }
let mut trace: int = 0
fn send() -> int {
    let x = perform Ask.ask(7)
    set trace (+ trace 1)
    return (+ x 10)
}
fn leave() -> int {
    let x = handle { (send) } with { ask n -> { return n } }
    set trace 100
    return x
}
fn resume_value() -> int {
    let x = handle { (send) } with { ask n -> { (+ n 1) } }
    return (+ x 100)
}
fn exercise() -> int {
    set trace 0
    let left = (leave)
    assert (== trace 0)
    let resumed = (resume_value)
    assert (== trace 1)
    return (+ left resumed)
}
shadow send { assert (== (exercise) 125) }
shadow leave { assert (== (exercise) 125) }
shadow resume_value { assert (== (exercise) 125) }
shadow exercise { assert (== (exercise) 125) }
fn main() -> int {
    assert (== (exercise) 125)
    (println "I dispatched the effect.")
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_byte_lexical_returns_preserve_once_only_effects(self):
        self.check_program('''enum Edge { Above = 256, High = 511 }
effect Ask { ask : int -> int }
let mut calls: int = 0
let mut continued: int = 0
fn tick() -> Edge { set calls (+ calls 1) return Edge.Above }
fn send() -> int {
    let value = perform Ask.ask(7)
    set continued (+ continued 1)
    return value
}
fn member() -> u8 {
    let value = handle { (send) } with { ask n -> { return Edge.High } }
    set continued (+ continued 10)
    return value
}
fn once() -> u8 {
    let value = handle { (send) } with { ask n -> { return (tick) } }
    set continued (+ continued 10)
    return value
}
fn exercise() -> int {
    set calls 0
    set continued 0
    assert (== (cast_int (member)) 255)
    assert (== calls 0)
    assert (== continued 0)
    assert (== (cast_int (once)) 0)
    assert (== calls 1)
    assert (== continued 0)
    return 0
}
shadow tick { assert (== (exercise) 0) }
shadow send { assert (== (exercise) 0) }
shadow member { assert (== (exercise) 0) }
shadow once { assert (== (exercise) 0) }
shadow exercise { assert (== (exercise) 0) }
fn main() -> int {
    assert (== (exercise) 0)
    (println "I dispatched the effect.")
    return 0
}
shadow main { assert (== (main) 0) }
''')

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
