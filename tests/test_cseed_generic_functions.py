"""I execute concrete generic bodies with mandatory source shadows."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CseedGenericFunctions(unittest.TestCase):
    def run_command(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def execute(self, source, files=None):
        with tempfile.TemporaryDirectory(prefix='nano-generic-') as tmp:
            work = Path(tmp)
            for name, contents in (files or {}).items():
                (work / name).write_text(contents)
            path, module, c, binary = [work / name for name in ('main.nano', 'main.nvm', 'main.c', 'main')]
            path.write_text(source)
            self.run_command([ROOT / 'bin/nano_virt', path, '--emit-nvm', '-o', module])
            self.run_command([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.run_command([ROOT / 'bin/nano_vm', module])
            self.run_command([ROOT / 'bin/nvm2c', module, '-o', c])
            cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc'))
            self.run_command(cc + ['-std=c11', '-Wall', '-Wextra', '-Werror',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                c, ROOT / 'bin/nano_aot_runtime.o', '-lm', '-o', binary])
            self.run_command([binary])

    def test_retained_record_result_regression(self):
        self.execute((ROOT / 'docs/evidence/generic-results-global-audit-20261010/generic-producer-repro.nano').read_text())

    def test_primitive_generic_fixture(self):
        self.execute((ROOT / 'tests/unit/test_generics.nano').read_text())

    def test_typed_locals_transitive_calls_and_recursion(self):
        self.execute('''struct Box { value: int }
fn identity(value: T) -> T { let copy: T = value return copy }
shadow identity { assert (== (identity 7) 7) }
fn forward(value: T) -> T { let copy: T = (identity value) return (identity copy) }
shadow forward { assert (== (forward true) true) }
fn repeat(value: T, count: int) -> T {
 if (== count 0) { return value }
 return (repeat (forward value) (- count 1))
}
shadow repeat { assert (== (repeat 5 3) 5) }
fn main() -> int {
 assert (== (repeat 17 5000) 17)
 assert (== (repeat 2.5 3) 2.5)
 assert (== (forward false) false)
 assert (== (repeat "text" 2) "text")
 let record: Box = (repeat Box { value: 29 } 3)
 assert (== record.value 29)
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_distinct_record_specializations(self):
        self.execute('''struct First { value: int }
struct Second { text: string }
fn identity(value: T) -> T { let copy: T = value return copy }
shadow identity { assert (== (identity 7) 7) }
fn main() -> int {
 let first: First = (identity First { value: 31 })
 let second: Second = (identity Second { text: "second" })
 assert (== first.value 31)
 assert (== second.text "second")
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_repeated_type_variable_rejects_distinct_records(self):
        with tempfile.TemporaryDirectory(prefix='nano-generic-refusal-') as tmp:
            path, module = Path(tmp) / 'main.nano', Path(tmp) / 'main.nvm'
            path.write_text('''struct First { value: int }
struct Second { value: int }
fn choose(a: T, b: T) -> T { return a }
shadow choose { assert (== (choose 1 2) 1) }
fn main() -> int {
 let result: First = (choose First { value: 1 } Second { value: 2 })
 return result.value
}
shadow main { assert true }
''')
            module.write_bytes(b'prior-output')
            result = subprocess.run([str(ROOT / 'bin/nano_virt'), str(path),
                '--emit-nvm', '-o', str(module)], cwd=ROOT, capture_output=True,
                text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('one concrete identity', result.stdout + result.stderr)
            self.assertEqual(module.read_bytes(), b'prior-output')

    def test_imported_qualified_and_selective_owners(self):
        self.execute('''module "a.nano" as first
from "b.nano" import choose as second
fn main() -> int {
 assert (== (first.choose 11 12) 11)
 assert (== (second 11 12) 12)
 assert (== (first.choose "left" "right") "left")
 return 0
}
shadow main { assert (== (main) 0) }
''', {'a.nano': 'pub fn choose(a: T, b: T) -> T { return a }\nshadow choose { assert (== (choose 1 2) 1) }\n',
      'b.nano': 'pub fn choose(a: T, b: T) -> T { return b }\nshadow choose { assert (== (choose 1 2) 2) }\n'})


if __name__ == '__main__':
    unittest.main()
