"""I retain effect-arm names without granting advisory metadata execution authority."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EffectBindingMetadata(unittest.TestCase):
    def checked(self, command, expected=0):
        result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result.stdout

    def produce(self, source, expected):
        with tempfile.TemporaryDirectory(prefix='nano-effect-names-') as directory:
            work = Path(directory)
            source_path, module = work/'source.nano', work/'source.nvm'
            source_path.write_text(source + '\nshadow main { assert (== (main) '+str(expected)+') }\n')
            self.checked([os.environ.get('NANOLANG_EFFECT_NAME_COMPILER', ROOT/'bin/nano_virt'),
                          source_path, '--emit-nvm', '-o', module])
            # The probe checks instruction boundaries, lookup endpoints, original-wire
            # equality and two canonical assembly cycles for every retained record.
            lines = self.checked([ROOT/'obj/test_local_bindings', module, '--require-debug'])
            bindings = [(function, int(slot), int(begin), int(end), name)
                        for function, slot, begin, end, name in
                        (line.split() for line in lines.splitlines())]
            self.checked([ROOT/'bin/nano_vm', module], expected)
            canonical = self.checked([ROOT/'bin/nanoisa', 'dump', module]).splitlines()
            strings = [line for line in canonical if line.startswith('.string ')]
            key = str(strings.index('.string "nano.local.v1"'))
            stripped = [line for line in canonical if not
                        (line.startswith('.metadata ') and line.split()[1] == key)]
            self.assertLess(len(stripped), len(canonical))
            assembly, unnamed = work/'unnamed.nasm', work/'unnamed.nvm'
            assembly.write_text('\n'.join(stripped)+'\n')
            self.checked([ROOT/'bin/nanoisa', 'asm', assembly, '-o', unnamed])
            self.checked([ROOT/'bin/nano_vm', unnamed], expected)
            self.checked([ROOT/'bin/nano_vm', '--verify-only', unnamed])
            return [binding for binding in bindings if binding[0] == 'main']

    def test_sibling_parameters_shadow_outer_binding(self):
        bindings = self.produce('''
effect Ask { ask : int -> int }
fn main() -> int {
    let value: int = 40
    let first: int = handle { perform Ask.ask(3) } with {
        ask value -> { let adjusted: int = (+ value 1) adjusted }
    }
    let second: int = handle { perform Ask.ask(2) } with {
        ask value -> { (+ value 1) }
    }
    assert (== value 40)
    return (+ first second)
}
''', 7)
        self.assertEqual(sorted(b[4] for b in bindings),
                         ['adjusted', 'first', 'second', 'value', 'value', 'value'])
        outer, first, second = sorted((b for b in bindings if b[4] == 'value'),
                                     key=lambda b: b[2])
        self.assertEqual(len({outer[1], first[1], second[1]}), 3)
        self.assertLess(outer[2], first[2])
        self.assertLess(first[2], first[3])
        self.assertLessEqual(first[3], second[2])
        self.assertGreaterEqual(outer[3], second[3])
        adjusted = next(b for b in bindings if b[4] == 'adjusted')
        self.assertLess(first[2], adjusted[2])
        self.assertLessEqual(adjusted[3], first[3])

    def test_nested_handlers_keep_distinct_parameters(self):
        bindings = self.produce('''
effect Ask { ask : int -> int }
fn main() -> int {
    let answer: int = handle { perform Ask.ask(4) } with {
        ask value -> {
            let inner: int = handle { perform Ask.ask(2) } with {
                ask value -> { (+ value 1) }
            }
            (+ value inner)
        }
    }
    return answer
}
''', 7)
        self.assertEqual(sorted(b[4] for b in bindings), ['answer', 'inner', 'value', 'value'])
        outer, inner = sorted((b for b in bindings if b[4] == 'value'), key=lambda b: b[2])
        self.assertNotEqual(outer[1], inner[1])
        self.assertLess(outer[2], inner[2])
        self.assertLess(inner[2], inner[3])
        self.assertLess(inner[3], outer[3])

    def test_typed_arguments_and_zero_argument_arm(self):
        bindings = self.produce('''
effect Calc { mix : int string bool -> int, tick : void -> int }
fn main() -> int {
    let left: int = handle { perform Calc.mix(3 "abc" true) } with {
        mix number text flag -> { assert flag (+ number (str_length text)) }
    }
    let right: int = handle { perform Calc.tick() } with { tick -> { 1 } }
    return (+ left right)
}
''', 7)
        names = [b[4] for b in bindings]
        self.assertEqual(sorted(names), ['flag', 'left', 'number', 'right', 'text'])
        arguments = [b for b in bindings if b[4] in ('number', 'text', 'flag')]
        self.assertEqual(len({b[1] for b in arguments}), 3)
        self.assertEqual(len({b[2:4] for b in arguments}), 1)
        self.assertLess(arguments[0][2], arguments[0][3])


    def test_lexical_return_retains_arm_interval(self):
        bindings = self.produce('''
effect Ask { ask : int -> int }
fn main() -> int {
    let answer: int = handle { perform Ask.ask(7) } with {
        ask value -> { return value }
    }
    return (+ answer 100)
}
''', 7)
        self.assertEqual(sorted(b[4] for b in bindings), ['answer', 'value'])
        parameter = next(b for b in bindings if b[4] == 'value')
        self.assertLess(parameter[2], parameter[3])


if __name__ == '__main__':
    unittest.main()
