"""I execute lexical binding isolation through both NanoISA producers."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT / 'bin/nanoc_stage2'))
TRANSLATOR = Path(os.environ.get('NANOLANG_TEST_NVM2C', ROOT / 'bin/nvm2c'))


class GenEnvScopeTests(unittest.TestCase):
    def checked(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180,
                                env={**os.environ, 'NANOLANG_ROOT': str(ROOT)})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def producers(self):
        return [('seed', ROOT / 'bin/nano_virt'), ('selfhost', COMPILER)]

    def execute(self, program):
        with tempfile.TemporaryDirectory(prefix='nano-lexical-scope-') as tmp:
            work = Path(tmp)
            source = work / 'input.nano'
            source.write_text(program)
            for name, producer in self.producers():
                with self.subTest(producer=name):
                    module, c, binary = [work / (name + suffix) for suffix in ('.nvm', '.c', '.exe')]
                    self.checked([producer, source, '--emit-nvm', '-o', module])
                    self.checked([ROOT / 'bin/nano_vm', '--verify-only', module])
                    self.assertEqual(self.checked([ROOT / 'bin/nano_vm', module]).stdout, 'scope:ok\n')
                    self.checked([TRANSLATOR, module, '-o', c])
                    cc = os.environ.get('NANO_NATIVE_TEST_CC') or shutil.which('cc')
                    self.assertTrue(cc)
                    self.checked([cc, '-std=c11', '-O0', '-Wall', '-Wextra', '-Werror',
                                  '-fsanitize=address,undefined', '-fno-sanitize-recover=all', c, '-lm', '-o', binary])
                    self.assertEqual(self.checked([binary]).stdout, 'scope:ok\n')

    def test_alias_types_mutability_and_global_bindings_restore(self):
        self.execute('''let mut count: int = 40
fn selected() -> int { return 17 }
shadow selected { assert (== (selected) 17) }
fn alternate() -> bool { return true }
shadow alternate { assert (alternate) }
fn check() -> int {
    set count 40
    let alias: fn() -> int = selected
    let mut value: int = 3
    if true {
        let selected: int = 9
        let alias: string = "inner"
        let value: bool = true
        let mut count: string = "local"
        set count "changed"
        assert (== selected 9)
        assert (== alias "inner")
        assert value
        assert (== count "changed")
    }
    if true {
        let selected: fn() -> bool = alternate
        assert (selected)
    }
    assert (== (selected) 17)
    assert (== (alias) 17)
    set value (+ value 1)
    set count (+ count value)
    assert (== value 4)
    return count
}
shadow check { assert (== (check) 44) }
fn main() -> int { assert (== (check) 44) (println "scope:ok") return 0 }
shadow main { assert (== (main) 0) }
''')

    def test_match_and_loop_binders_restore_outer_names(self):
        self.execute('''union Choice { Number { value: int }, Text { value: string } }
struct Item { value: int }
fn check(choice: Choice) -> int {
    let item: string = "outside"
    let mut total: int = 0
    match choice {
        Number(item) => { set total item.value }
        Text(item) => { set total (str_length item.value) }
    }
    assert (== item "outside")
    for item in (range 0 3) { set total (+ total item) }
    assert (== item "outside")
    for item in [4, 5] { set total (+ total item) }
    assert (== item "outside")
    let records: array<Item> = [Item { value: 6 }]
    for item in records { set total (+ total item.value) }
    assert (== item "outside")
    return total
}
shadow check {
    assert (== (check Choice.Number { value: 7 }) 25)
    assert (== (check Choice.Text { value: "abc" }) 21)
}
fn main() -> int {
    assert (== (check Choice.Number { value: 7 }) 25)
    assert (== (check Choice.Text { value: "abc" }) 21)
    (println "scope:ok")
    return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_out_of_scope_and_immutable_writes_preserve_output(self):
        cases = {
            'block escape': 'if true { let hidden: int = 3 } return hidden',
            'range escape': 'for hidden in (range 0 1) {} return hidden',
            'array escape': 'for hidden in [1] {} return hidden',
            'immutable inner': 'let mut value: int = 1 if true { let value: int = 2 set value 3 } return value',
            'immutable outer': 'let value: int = 1 if true { let mut value: int = 2 set value 3 } set value 4 return value',
            'shadowed function': 'let selected: int = 2 return (selected)',
        }
        with tempfile.TemporaryDirectory(prefix='nano-lexical-refusals-') as tmp:
            source, output = Path(tmp) / 'input.nano', Path(tmp) / 'output.nvm'
            for label, body in cases.items():
                source.write_text('fn selected() -> int { return 17 }\nshadow selected { assert (== (selected) 17) }\n'
                                  'fn main() -> int { ' + body + ' }\nshadow main { assert true }\n')
                for name, producer in self.producers():
                    with self.subTest(case=label, producer=name):
                        output.write_bytes(b'retained module')
                        result = subprocess.run([producer, source, '--emit-nvm', '-o', output],
                                                cwd=ROOT, capture_output=True, text=True, timeout=180,
                                                env={**os.environ, 'NANOLANG_ROOT': str(ROOT)})
                        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                        self.assertEqual(output.read_bytes(), b'retained module')


if __name__ == '__main__':
    unittest.main()
