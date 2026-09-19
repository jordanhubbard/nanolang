"""I keep declared push identity distinct from my mutable-array builtin."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(os.environ.get('NANO_MUTATION_TEST_ROOT', Path(__file__).resolve().parents[1]))


class DeclaredArrayPushIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nanolang-declared-push-'))
        print('I retain declared push artifacts in ' + str(cls.work), flush=True)

    def command(self, args, success=True):
        result = subprocess.run([str(arg) for arg in args], cwd=ROOT,
                                capture_output=True, text=True, timeout=180)
        log = self.work / ('command-%04d.log' % len(list(self.work.glob('command-*.log'))))
        log.write_text(repr([str(arg) for arg in args]) + '\n' + result.stdout + result.stderr)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
        return result

    def positive(self, label, text, canonical=True):
        source = self.work / (label + '.nano')
        source.write_text(text)
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(case=label, compiler=compiler):
                binary = self.work / (label + '-' + compiler)
                self.command([ROOT / 'bin' / compiler, source, '-o', binary])
                self.command([binary])
        if canonical:
            dumps = []
            for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
                with self.subTest(case=label, compiler=compiler, route='canonical'):
                    output = self.work / (label + '-' + compiler + '.nvm')
                    self.command([ROOT / 'bin' / compiler, source, '--emit-nvm', '-o', output])
                    self.command([ROOT / 'bin/nano_vm', '--verify-only', output])
                    self.command([ROOT / 'bin/nano_vm', output])
                    dumps.append(self.command([ROOT / 'bin/nanoisa', 'dump', output]).stdout)
            self.assertEqual(dumps[0], dumps[1])
            self.assertEqual(dumps[0], dumps[2])

    def test_declared_scalar_order_and_initializer(self):
        self.positive('declared-order', '''fn first(state: array<int>) -> int {
 assert (== (at state 0) 0) (array_set state 0 1) return 11
}
shadow first { let state: array<int> = [0] assert (== (first state) 11) assert (== (at state 0) 1) }
fn second(state: array<int>) -> int {
 assert (== (at state 0) 1) (array_set state 0 2) return 22
}
shadow second { let state: array<int> = [1] assert (== (second state) 22) assert (== (at state 0) 2) }
fn array_push(left: int, right: int) -> int { assert (== left 11) assert (== right 22) return (+ left right) }
shadow array_push { assert (== (array_push 11 22) 33) }
fn main() -> int {
 let state: array<int> = [0]
 let array_push: int = (array_push (first state) (second state))
 assert (== array_push 33) assert (== (at state 0) 2) return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_declared_array_return_element_facts(self):
        self.positive('declared-array', '''fn array_push(values: array<int>, value: float) -> array<float> {
 assert (== (at values 0) 9) return [value]
}
shadow array_push { let result: array<float> = (array_push [9] 2.5) assert (== (at result 0) 2.5) }
fn main() -> int {
 let result: array<float> = (array_push [9] 3.5)
 assert (== (at result 0) 3.5) assert (== (array_length result) 1) return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_lexical_signature_and_restoration(self):
        self.positive('lexical', '''fn array_push(left: int, right: int) -> int { return (+ left right) }
shadow array_push { assert (== (array_push 4 5) 9) }
fn selected(left: int, right: int) -> bool { return (== left right) }
shadow selected { assert (selected 7 7) assert (not (selected 7 8)) }
fn main() -> int {
 if true { let array_push: fn(int, int) -> bool = selected assert (array_push 7 7) }
 assert (== (array_push 4 5) 9) return 0
}
shadow main { assert (== (main) 0) }
''', canonical=False)  # I do not expand canonical indirect-call admission.
        self.positive('declared-function-value', '''fn array_push(left: int, right: int) -> int { return (+ left right) }
shadow array_push { assert (== (array_push 4 5) 9) }
fn main() -> int {
 let selected: fn(int, int) -> int = array_push
 assert (== (selected 4 5) 9)
 if true { let array_push: fn(int, int) -> int = selected assert (== (array_push 7 8) 15) }
 assert (== (array_push 1 2) 3) return 0
}
shadow main { assert (== (main) 0) }
''', canonical=False)

    def test_unbound_push_alias_and_empty(self):
        self.positive('builtin', '''fn main() -> int {
 let values: array<float> = []
 let alias: array<float> = (array_push values 2.5)
 (array_push alias 3.5)
 assert (== (array_length values) 2)
 assert (== (at values 0) 2.5) assert (== (at alias 1) 3.5) return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_refusals_preserve_publication(self):
        base = 'fn array_push(a:int,b:int)->int{return (+ a b)} shadow array_push{assert (== (array_push 1 2) 3)}\n'
        cases = {
            'arity': base + 'fn main()->int{return (array_push 1)} shadow main{assert (== (main) 0)}',
            'type': base + 'fn main()->int{return (array_push 1 true)} shadow main{assert (== (main) 0)}',
            'false-shadow': base.replace('(== (array_push 1 2) 3)', '(== (array_push 1 2) 4)') + 'fn main()->int{return 0} shadow main{assert (== (main) 0)}',
        }
        for label, text in cases.items():
            source = self.work / (label + '.nano')
            source.write_text(text)
            for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2', 'nano_virt'):
                flags = ['--emit-nvm'] if compiler == 'nano_virt' else []
                output = self.work / (label + '-' + compiler + '.prior')
                output.write_bytes(b'prior verified output')
                self.command([ROOT / 'bin' / compiler, source, *flags, '-o', output], False)
                self.assertEqual(output.read_bytes(), b'prior verified output')
        for name in ('array_set', 'array_length'):
            source = self.work / (name + '-reserved.nano')
            source.write_text(f'fn {name}(value:int)->int{{return value}} shadow {name}{{assert (== ({name} 7) 7)}} fn main()->int{{return 0}} shadow main{{assert (== (main) 0)}}')
            for compiler in ('nanoc_c', 'nano_virt'):
                output = self.work / (name + '-' + compiler + '.prior')
                output.write_bytes(b'prior verified output')
                flags = ['--emit-nvm'] if compiler == 'nano_virt' else []
                result = self.command([ROOT / 'bin' / compiler, source, *flags, '-o', output], False)
                self.assertEqual(output.read_bytes(), b'prior verified output')
                self.assertRegex(result.stdout + result.stderr, r'(?i)reserved|built.?in')


if __name__ == '__main__':
    unittest.main()
