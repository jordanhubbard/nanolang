"""I execute self-hosted function values in NanoVM; native parity has its own gate."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests import test_selfhost_returned_calls as returned_calls

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT / 'bin/nanoc_stage2')).resolve()
VM = ROOT / 'bin/nano_vm'


class VMReturnedCalls(returned_calls.ReturnedCalls):
    # I reuse the original execution/ordering/refusal assertions with an explicit
    # VM launcher. This does not replace or skip the original native tests.
    def compile(self, source, directory):
        module, launcher = directory / 'program.nvm', directory / 'run-vm'
        result = subprocess.run([COMPILER, source, '--emit-nvm', '-o', module],
                                cwd=ROOT, capture_output=True, timeout=120)
        if result.returncode == 0:
            check = subprocess.run([VM, '--verify-only', module], capture_output=True, timeout=15)
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
            launcher.write_text('#!/bin/sh\nexec ' + shlex.join([str(VM), str(module)]) + '\n')
            launcher.chmod(0o755)
        return result, launcher

    def execute_source(self, text, expected):
        with tempfile.TemporaryDirectory(prefix='nano-function-values-') as tmp:
            directory = Path(tmp)
            source = directory / 'input.nano'
            source.write_text(text)
            result, launcher = self.compile(source, directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([launcher], capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout, expected)

    def test_mutating_argument_does_not_replace_saved_callee(self):
        self.execute_source('''
fn first(n:int)->int{return (+ n 10)}
shadow first { assert (== (first 2) 12) }
fn second(n:int)->int{return (+ n 20)}
shadow second { assert (== (second 2) 22) }
let mut selected: fn(int)->int = first
fn argument()->int { set selected second (println "argument") return 2 }
shadow argument { assert (== (argument) 2) }
fn main()->int {
    set selected first
    assert (== (selected (argument)) 12)
    assert (== (selected 2) 22)
    return 0
}
shadow main { assert (== (main) 0) }
''', b'argument\n')

    def test_void_and_function_valued_results_keep_stack_counts(self):
        self.execute_source('''
fn sink(n:int)->void { (println n) }
shadow sink { (sink 7) }
fn select()->fn(int)->void { return sink }
shadow select { ((select) 7) }
fn relay(f:fn(int)->void)->fn(int)->void { return f }
shadow relay { ((relay sink) 7) }
fn invoke(f:fn(int)->void,n:int)->void { (f n) }
shadow invoke { (invoke sink 7) }
fn main()->int { (invoke (relay (select)) 42) ((select) 9) return 0 }
shadow main { assert (== (main) 0) }
''', b'42\n9\n')

    def test_contextual_bytes_arrays_and_record_arguments(self):
        self.execute_source('''
struct Point { x:int }
fn byte(n:u8)->int { return (cast_int n) }
shadow byte { assert (== (byte 7) 7) }
fn length(values:array<string>)->int { return (array_length values) }
shadow length { assert (== (length []) 0) }
fn increment(point:Point)->Point { return Point { x: (+ point.x 1) } }
shadow increment { assert (== (increment (Point { x: 3 })).x 4) }
fn main()->int {
    let b:fn(u8)->int = byte
    let l:fn(array<string>)->int = length
    let r:fn(Point)->Point = increment
    assert (== (b 255) 255)
    assert (== (l []) 0)
    assert (== (l ["a", "b"]) 2)
    let point:Point = (r (Point { x: 41 }))
    (println point.x)
    return 0
}
shadow main { assert (== (main) 0) }
''', b'42\n')

    def test_wrong_signature_or_non_callable_preserves_prior_module(self):
        cases = [
            'let f:fn(string)->int = identity return 0',
            'let f:int = 0 return (f 42)',
            'let f:fn(int)->int = identity return (f true)',
            'let f:fn(int)->int = identity return (f 1 2)',
        ]
        for statement in cases:
            with self.subTest(statement=statement), tempfile.TemporaryDirectory(prefix='nano-function-refusal-') as tmp:
                directory = Path(tmp)
                source, output = directory / 'input.nano', directory / 'prior.nvm'
                source.write_text('fn identity(n:int)->int{return n}\n'
                                  'shadow identity { assert (== (identity 7) 7) }\n'
                                  'fn main()->int{' + statement + '}\nshadow main { assert true }\n')
                output.write_bytes(b'prior module\x00')
                result = subprocess.run([COMPILER, source, '--emit-nvm', '-o', output],
                                        cwd=ROOT, capture_output=True, timeout=120)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), b'prior module\x00')

    def test_imported_function_value_uses_resolved_alias(self):
        with tempfile.TemporaryDirectory(prefix='nano-function-import-') as tmp:
            directory = Path(tmp)
            dependency = directory / 'values.nano'
            dependency.write_text('pub fn answer(n:int)->int{return (+ n 1)}\n'
                                  'shadow answer {assert (== (answer 41) 42)}\n')
            source = directory / 'input.nano'
            source.write_text('from "' + str(dependency) + '" import answer as chosen\n'
                              'fn main()->int{let f:fn(int)->int = chosen (println (f 41)) return 0}\n'
                              'shadow main {assert (== (main) 0)}\n')
            result, launcher = self.compile(source, directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([launcher], capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout, b'42\n')


if __name__ == '__main__':
    unittest.main()
