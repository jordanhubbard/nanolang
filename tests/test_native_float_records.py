"""I retain float fields through source and bytecode aggregate transport."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NativeFloatRecords(unittest.TestCase):
    def checked(self, args):
        result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
            text=True, timeout=120, env={**os.environ,
            'ASAN_OPTIONS': 'detect_leaks=0' if sys.platform == 'darwin' else 'detect_leaks=1'})
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result.stdout

    def execute(self, work, module, expected):
        self.checked([ROOT/'bin/nano_vm', '--verify-only', module])
        self.assertEqual(self.checked([ROOT/'bin/nano_vm', module]), expected)
        source, binary = work/'native.c', work/'native'
        self.checked([ROOT/'bin/nvm2c', module, '-o', source])
        compiler = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc'))
        self.checked([*compiler, '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
            '-fsanitize=address,undefined', '-fno-sanitize-recover=all', source, '-lm', '-o', binary])
        self.assertEqual(self.checked([binary]), expected)

    def paired(self, text, expected=''):
        with tempfile.TemporaryDirectory(prefix='nano-float-record-') as tmp:
            work = Path(tmp); source = work/'input.nano'; source.write_text(text)
            for producer in ('nano_virt', 'nanoisa_emit'):
                with self.subTest(producer=producer):
                    module = work/(producer+'.nvm')
                    self.checked([ROOT/'bin'/producer, source, '--emit-nvm', '-o', module])
                    self.execute(work, module, expected)

    def test_nested_copies_replacement_calls_and_record_arrays(self):
        self.paired('''struct Sample { value: float, label: string, count: int, valid: bool }
struct Nest { child: Sample }
fn read(sample: Sample) -> float { return sample.value }
shadow read {
 let sample: Sample = Sample { value: 1.25, label: "kept", count: 7, valid: true }
 assert (== (read sample) 1.25)
}
fn update(input: Nest) -> Nest {
 let mut output: Nest = input
 let child: Sample = input.child
 set output Nest { child: Sample { value: -2.5, label: child.label, count: child.count, valid: child.valid } }
 return output
}
shadow update {
 let input: Nest = Nest { child: Sample { value: 1.25, label: "kept", count: 7, valid: true } }
 let output: Nest = (update input)
 assert (== output.child.value -2.5)
 assert (== input.child.value 1.25)
}
fn main() -> int {
 let original: Sample = Sample { value: 1.25, label: "kept", count: 7, valid: true }
 let wrapped: Nest = Nest { child: original }
 let changed: Nest = (update wrapped)
 assert (== (read original) 1.25)
 assert (== wrapped.child.value 1.25)
 assert (== changed.child.value -2.5)
 assert (== changed.child.label "kept")
 assert (== changed.child.count 7)
 assert changed.child.valid
 let rows: array<Sample> = [original, changed.child]
 let first: Sample = (at rows 0)
 let second: Sample = (at rows 1)
 assert (== (read first) 1.25)
 assert (== (read second) -2.5)
 return 0
}
shadow main { assert true }
''')

    def test_tagged_float_global_packed_into_a_record(self):
        self.paired('''struct Input { value: float }
let mut angle: float = 2.5
fn read(input: Input) -> float { return (+ input.value 1.0) }
shadow read {
 let input: Input = Input { value: 2.5 }
 assert (== (read input) 3.5)
}
fn main() -> int {
 let input: Input = Input { value: angle }
 set angle -7.5
 assert (== (read input) 3.5)
 let second: Input = Input { value: angle }
 assert (== (read second) -6.5)
 return 0
}
shadow main { assert true }
''')

    def test_signed_zero_nan_and_infinities_in_aggregate_cells(self):
        body = ''
        for value in ('-0', 'inf', '-inf', 'nan'):
            body += f'PUSH_F64 {value}\nAGG_PACK 0 0 0 1\nAGG_PACK 0 1 0 1\n'
            body += 'ARR_LITERAL 8 1\nPUSH_I64 0\nARR_GET\nAGG_GET 0\nAGG_GET 0\n'
            body += 'DUP\nTYPE_CHECK 3\nASSERT\n'
            if value == 'nan':
                body += 'DUP\nF64_NE\nASSERT\n'
            else:
                body += 'CAST_STRING\nPRINTLN\n'
        with tempfile.TemporaryDirectory(prefix='nano-float-cell-') as tmp:
            work = Path(tmp); source = work/'input.nasm'; module = work/'input.nvm'
            source.write_text('.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')
            self.checked([ROOT/'bin/nanoisa', 'asm', source, '-o', module])
            self.execute(work, module, '-0\ninf\n-inf\n')

if __name__ == '__main__':
    unittest.main()
