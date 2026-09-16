"""I exercise native handler captures, nesting, and dynamic frame lifetimes."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NativeEffectExecution(unittest.TestCase):
    def run_program(self, body, expected=0, foreign=False):
        with tempfile.TemporaryDirectory(prefix="nano-native-effects-") as tmp:
            source = Path(tmp) / "effects.nano"
            output = Path(tmp) / "effects"
            source.write_text(body)
            environment = os.environ.copy()
            if foreign:
                header = Path(tmp) / "foreign.h"
                header.write_text("static inline void nl_test_invoke(void (*callback)(void)) { callback(); }\n" if foreign is True else foreign)
                environment["NANO_CFLAGS"] = environment.get("NANO_CFLAGS", "") + " -D_GNU_SOURCE -include " + str(header)
            result = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source), "-o", str(output)], cwd=ROOT, env=environment, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            run = subprocess.run([str(output)], cwd=ROOT, capture_output=True, timeout=10)
            if expected == 0:
                self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))
            else:
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(b"foreign callback boundary" if foreign else b"I cannot perform unhandled effect Ask.ask.", run.stderr)

    def test_owned_locals_release_during_nonlocal_return(self):
        with tempfile.TemporaryDirectory(prefix="nano-effect-cleanup-") as tmp:
            output = Path(tmp) / "cleanup"
            command = [*shlex.split(os.environ.get("CC", "cc")), "-std=gnu11", "-I", str(ROOT / "src"),
                       *shlex.split(os.environ.get("NANO_CFLAGS", "")),
                       str(ROOT / "tests/test_native_effect_runtime.c"),
                       *[str(ROOT / "src/runtime" / name) for name in
                         ("effect_runtime.c", "gc.c", "dyn_array.c", "gc_struct.c")], "-o", str(output)]
            built = subprocess.run(command, capture_output=True, timeout=30)
            self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace"))
            run = subprocess.run([str(output)], capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))

    def test_imported_perform_reaches_callers_handler(self):
        with tempfile.TemporaryDirectory(prefix="nano-imported-effects-") as tmp:
            directory = Path(tmp)
            (directory / "operations.nano").write_text('''
effect Ask { ask : int -> int }
pub fn send(value: int) -> int { return perform Ask.ask(value) }
shadow send { assert true }
''')
            source = directory / "main.nano"
            source.write_text('''
module "operations.nano" as Ops
fn main() -> int {
 let result = handle { (Ops.send 4) } with { ask n -> { (+ n 8) } }
 assert (== result 12)
 return 0
}
shadow main { assert true }
''')
            output = directory / "program"
            built = subprocess.run([str(ROOT / "bin/nanoc_c"), str(source), "-o", str(output)], cwd=ROOT, capture_output=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr.decode(errors="replace"))
            run = subprocess.run([str(output)], capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))

    def test_mutable_lexical_capture_and_perform_value(self):
        self.run_program('''
effect Ask { ask : int -> int }
fn exercise() -> int {
 let mut captured: int = 3
 let first = handle { perform Ask.ask(7) } with { ask n -> { set captured (+ captured n) captured } }
 let second = handle { perform Ask.ask(2) } with { ask n -> { (+ n captured) } }
 return (+ first second)
}
shadow exercise { assert (== (exercise) 22) }
fn main() -> int { assert (== (exercise) 22) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_nested_dynamic_handler_restores_outer(self):
        self.run_program('''
effect Ask { ask : int -> int }
fn nested() -> int {
 let inside = handle { perform Ask.ask(2) } with { ask n -> { (+ n 10) } }
 let outside = perform Ask.ask(3)
 return (+ inside outside)
}
fn exercise() -> int { return handle { (nested) } with { ask n -> { (+ n 100) } } }
shadow nested { assert (== (exercise) 115) }
shadow exercise { assert (== (exercise) 115) }
fn main() -> int { assert (== (exercise) 115) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_handler_installs_nested_handler_with_lexical_return(self):
        self.run_program(''' 
effect Ask { ask : int -> int }
effect Other { other : int -> int }
fn exercise() -> int {
 let mut value: int = 1
 let ignored = handle { perform Ask.ask(7) } with {
  ask n -> {
   let inner = handle { perform Other.other(n) } with { other k -> { set value (+ value k) return value } }
   return 99
  }
 }
 return 100
}
shadow exercise { assert (== (exercise) 8) }
fn main() -> int { assert (== (exercise) 8) return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_foreign_callback_cannot_escape_its_native_activation(self):
        self.run_program('''
effect Ask { ask : int -> int }
extern fn nl_test_invoke(callback: fn() -> void) -> void
fn callback() -> void { let ignored = perform Ask.ask(1) }
shadow callback { assert true }
fn main() -> int {
 unsafe {
  let ignored = handle { (nl_test_invoke callback) } with { ask n -> { return n } }
 }
 return 0
}
shadow main { assert true }
''', expected=1, foreign=True)

    def test_foreign_callback_can_resume_without_nonlocal_return(self):
        self.run_program('''
effect Ask { ask : int -> int }
let mut observed: int = 0
extern fn nl_test_invoke(callback: fn() -> void) -> void
fn callback() -> void { let ignored = perform Ask.ask(7) }
shadow callback { assert true }
fn main() -> int {
 unsafe {
  let ignored = handle { (nl_test_invoke callback) } with { ask n -> { set observed n n } }
 }
 assert (== observed 7)
 return 0
}
shadow main { assert true }
''', foreign=True)

    def test_generated_owned_local_is_released_on_handler_return(self):
        self.run_program('''
effect Exit { leave : void -> void }
opaque type Owned
extern fn nl_owned_create() -> Owned
extern fn nl_owned_count() -> int
fn send() -> void { unsafe { let item: Owned = (nl_owned_create) perform Exit.leave() } }
shadow send { assert true }
fn leave() -> int {
 let ignored = handle { (send) } with { leave -> { return 42 } }
 return 99
}
shadow leave { assert true }
fn main() -> int {
 let mut i: int = 0
 while (< i 100) { assert (== (leave) 42) set i (+ i 1) }
 unsafe { assert (== (nl_owned_count) 100) }
 return 0
}
shadow main { assert true }
''', foreign='''
#include "runtime/gc.h"
static int64_t finalized;
static void finish_owned(void *value) { (void)value; ++finalized; }
static inline void *nl_owned_create(void) { return gc_alloc_opaque(8, finish_owned); }
static inline int64_t nl_owned_count(void) { return finalized; }
''')

    def test_handler_return_transfers_captured_owned_value(self):
        self.run_program('''
effect Exit { leave : void -> void }
opaque type Owned
extern fn nl_owned_create() -> Owned
extern fn nl_owned_count() -> int
fn leave() -> Owned {
 unsafe {
  let item: Owned = (nl_owned_create)
  let ignored = handle { perform Exit.leave() } with { leave -> { return item } }
  return item
 }
}
shadow leave { assert true }
fn exercise() -> void {
 let item: Owned = (leave)
 unsafe { assert (== (nl_owned_count) 0) }
}
shadow exercise { assert true }
fn main() -> int {
 (exercise)
 unsafe { assert (== (nl_owned_count) 1) }
 return 0
}
shadow main { assert true }
''', foreign='''
#include "runtime/gc.h"
static int64_t finalized;
static void finish_owned(void *value) { (void)value; ++finalized; }
static inline void *nl_owned_create(void) { return gc_alloc_opaque(8, finish_owned); }
static inline int64_t nl_owned_count(void) { return finalized; }
''')

    def test_return_removes_handler(self):
        self.run_program('''
effect Ask { ask : int -> int }
fn leave() -> int {
 let ignored = handle { perform Ask.ask(7) } with { ask n -> { return n } }
 return 99
}
shadow leave { assert (== (leave) 7) }
fn main() -> int { assert (== (leave) 7) let value = perform Ask.ask(1) return value }
shadow main { assert true }
''', expected=1)

if __name__ == "__main__":
    unittest.main()
