"""I retain concrete callback annotations across my compiler boundaries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILERS = os.environ.get('NANO_CALLBACK_COMPILERS', 'nanoc_c,nanoc_stage1,nanoc_stage2').split(',')
PRELUDE = '''union Box<T> { Some { value: T } }
fn make_box() -> Box<int> { return Box.Some { value: 7 } }
shadow make_box { let box: Box<int> = (make_box) match box { Some(payload) => { assert (== payload.value 7) } } }
fn read_box(box: Box<int>) -> int { match box { Some(payload) => { return payload.value } } }
shadow read_box { assert (== (read_box (make_box)) 7) }
'''

class GenericFunctionValues(unittest.TestCase):
    def check(self, body, reject=False):
        source = PRELUDE + body + '\nshadow main { assert (== (main) 0) }\n'
        for compiler in COMPILERS:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-callback-') as directory:
                path, output = Path(directory)/'input.nano', Path(directory)/'output'
                path.write_text(source)
                output.write_text('prior artifact')
                result = subprocess.run([ROOT/'bin'/compiler, path, '-o', output], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                diagnostic = result.stdout + result.stderr
                if reject:
                    self.assertNotEqual(result.returncode, 0, diagnostic)
                    self.assertEqual(output.read_text(), 'prior artifact')
                    self.assertNotIn('C compilation failed', diagnostic)
                else:
                    self.assertEqual(result.returncode, 0, diagnostic)
                    result = subprocess.run([output], capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generic_result(self):
        self.check('''fn apply(make: fn() -> Box<int>) -> int { return (read_box (make)) }
shadow apply { assert (== (apply make_box) 7) }
fn main() -> int { return (- (apply make_box) 7) }''')

    def test_generic_parameter(self):
        self.check('''fn apply(read: fn(Box<int>) -> int) -> int { return (read (make_box)) }
shadow apply { assert (== (apply read_box) 7) }
fn main() -> int { return (- (apply read_box) 7) }''')

    def test_local_function_value(self):
        self.check('''fn main() -> int {
 let make: fn() -> Box<int> = make_box
 return (- (read_box (make)) 7)
}''')

    def test_forwarded_function_value(self):
        self.check('''fn apply(make: fn() -> Box<int>) -> int { return (read_box (make)) }
shadow apply { assert (== (apply make_box) 7) }
fn forward(make: fn() -> Box<int>) -> int { return (apply make) }
shadow forward { assert (== (forward make_box) 7) }
fn main() -> int { return (- (forward make_box) 7) }''')

    def test_reject_wrong_generic_result(self):
        self.check('''fn apply(make: fn() -> Box<bool>) -> int { return 0 }
shadow apply { assert true }
fn main() -> int { return (apply make_box) }''', reject=True)

    def test_reject_wrong_local_generic_result(self):
        self.check('''fn main() -> int { let make: fn() -> Box<bool> = make_box return 0 }''', reject=True)

    def test_reject_wrong_generic_parameter(self):
        self.check('''fn apply(read: fn(Box<bool>) -> int) -> int { return 0 }
shadow apply { assert true }
fn main() -> int { return (apply read_box) }''', reject=True)

    def test_reject_wrong_forwarded_signature(self):
        self.check('''fn apply(make: fn() -> Box<bool>) -> int { return 0 }
shadow apply { assert true }
fn forward(make: fn() -> Box<int>) -> int { return (apply make) }
shadow forward { assert true }
fn main() -> int { return (forward make_box) }''', reject=True)

    def test_nested_array_callback(self):
        self.check('''fn identity(value: array<array<int>>) -> array<array<int>> { return value }
shadow identity { assert (== (array_length (identity [[7]])) 1) }
fn apply(f: fn(array<array<int>>) -> array<array<int>>) -> int {
 let result: array<array<int>> = (f [[7]])
 return (at (at result 0) 0)
}
shadow apply { assert (== (apply identity) 7) }
fn main() -> int { return (- (apply identity) 7) }''')

    def test_reject_nested_array_signature(self):
        self.check('''fn identity(value: array<array<string>>) -> array<array<string>> { return value }
shadow identity { assert (== (array_length (identity [["value"]])) 1) }
fn apply(f: fn(array<array<int>>) -> array<array<int>>) -> int { return 0 }
shadow apply { assert true }
fn main() -> int { return (apply identity) }''', reject=True)

    def test_reject_wrong_indirect_generic_argument(self):
        self.check('''fn read_bool(box: Box<bool>) -> int {
 match box { Some(payload) => { if payload.value { return 1 } else { return 0 } } }
}
shadow read_bool { let value: Box<bool> = Box.Some { value: true } assert (== (read_bool value) 1) }
fn apply(read: fn(Box<bool>) -> int) -> int { return (read (make_box)) }
shadow apply { assert true }
fn main() -> int { return (apply read_bool) }''', reject=True)

    def test_vm_generic_callback_context(self):
        source = PRELUDE + '''fn apply(make: fn() -> Box<int>) -> int { return (read_box (make)) }
shadow apply { assert (== (apply make_box) 7) }
fn main() -> int { return (- (apply make_box) 7) }
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-vm-callback-') as directory:
            path, output = Path(directory)/'input.nano', Path(directory)/'output.nvm'
            path.write_text(source)
            result = subprocess.run([ROOT/'bin/nano_virt', path, '--emit-nvm', '-o', output],
                                    cwd=ROOT, capture_output=True, text=True, timeout=120)
            diagnostic = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, diagnostic)
            self.assertNotIn('cannot determine this function value', diagnostic)
            result = subprocess.run([ROOT/'bin/nano_vm', output], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_reject_wrong_indirect_nested_array_argument(self):
        self.check('''fn read(values: array<array<int>>) -> int { return 0 }
shadow read { assert true }
fn apply(f: fn(array<array<int>>) -> int) -> int { return (f [[true]]) }
shadow apply { assert true }
fn main() -> int { return (apply read) }''', reject=True)

    def test_computed_generic_callback(self):
        self.check('''fn choose() -> fn(Box<int>) -> int { return read_box }
shadow choose { assert (== ((choose) (make_box)) 7) }
fn main() -> int { return (- ((choose) (make_box)) 7) }''')

    def test_reject_computed_nested_array_argument(self):
        self.check('''fn read(values: array<array<int>>) -> int { return 0 }
shadow read { assert true }
fn choose() -> fn(array<array<int>>) -> int { return read }
shadow choose { assert true }
fn main() -> int { return ((choose) [[true]]) }''', reject=True)

    def test_indirect_record_literal_identity(self):
        self.check('''struct First { value: int }
fn read(value: First) -> int { return value.value }
shadow read { assert (== (read First { value: 7 }) 7) }
fn apply(f: fn(First) -> int) -> int { return (f First { value: 7 }) }
shadow apply { assert (== (apply read) 7) }
fn main() -> int { return (- (apply read) 7) }''')

    def test_reject_indirect_wrong_record_literal(self):
        self.check('''struct First { value: int }
struct Second { value: int }
fn read(value: First) -> int { return value.value }
shadow read { assert (== (read First { value: 7 }) 7) }
fn apply(f: fn(First) -> int) -> int { return (f Second { value: 7 }) }
shadow apply { assert true }
fn main() -> int { return (- (apply read) 7) }''', reject=True)

    def test_qualified_import_callback_signatures(self):
        module = '''pub fn apply(f: fn(array<array<int>>) -> array<array<int>>, values: array<array<int>>) -> array<array<int>> { return (f values) }
shadow apply { assert true }
'''
        for compiler in COMPILERS:
            for mismatch in (False, True):
                with self.subTest(compiler=compiler, mismatch=mismatch), tempfile.TemporaryDirectory(prefix='nano-qualified-callback-') as directory:
                    directory = Path(directory)
                    (directory/'callbacks.nano').write_text(module)
                    element, literal = ('string', '"value"') if mismatch else ('int', '7')
                    source = f'''import "callbacks.nano" as cb
fn identity(values: array<array<{element}>>) -> array<array<{element}>> {{ return values }}
shadow identity {{ assert (== (array_length (identity [[{literal}]])) 1) }}
fn main() -> int {{
 let result: array<array<int>> = (cb.apply identity [[7]])
 return (- (at (at result 0) 0) 7)
}}
shadow main {{ assert (== (main) 0) }}
'''
                    path, output = directory/'main.nano', directory/'output'
                    path.write_text(source)
                    output.write_text('prior artifact')
                    result = subprocess.run([ROOT/'bin'/compiler, path, '-o', output], cwd=ROOT,
                                            capture_output=True, text=True, timeout=120)
                    diagnostic = result.stdout + result.stderr
                    if mismatch:
                        self.assertNotEqual(result.returncode, 0, diagnostic)
                        self.assertEqual(output.read_text(), 'prior artifact')
                        self.assertNotIn('C compilation failed', diagnostic)
                    else:
                        self.assertEqual(result.returncode, 0, diagnostic)
                        result = subprocess.run([output], capture_output=True, text=True, timeout=15)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

if __name__ == '__main__': unittest.main()
