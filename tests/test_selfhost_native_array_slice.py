"""I qualify the missing native helper with real source and generated C."""
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    'original': '''let values: array<int> = [10, 20, 30]
let part: array<int> = (array_slice values 0 2)
assert (== (array_length part) 2)
assert (== (at part 1) 20)''',
    'bounds': '''let values: array<int> = [10, 20, 30]
assert (== (array_length (array_slice values -1 2)) 2)
assert (== (array_length (array_slice values 1 -1)) 0)
assert (== (array_length (array_slice values 9223372036854775807 2)) 0)
assert (== (array_length (array_slice values 1 9223372036854775807)) 2)
assert (== (array_length (array_slice values 3 0)) 0)
let empty: array<int> = []
assert (== (array_length (array_slice empty 0 5)) 0)
let copy: array<int> = (array_slice values 0 3)
(array_set copy 1 99)
assert (== (at copy 1) 99)
assert (== (at values 1) 20)''',
    'leaves': '''let flags: array<bool> = [true, false, true]
let bs: array<bool> = (array_slice flags 1 2)
assert (not (at bs 0))
assert (at bs 1)
let words: array<string> = ["one", "two", "three"]
let ss: array<string> = (array_slice words 1 2)
assert (== (at ss 0) "two")
let numbers: array<float> = [1.25, -2.5, 3.75]
let fs: array<float> = (array_slice numbers 1 2)
assert (== (at fs 0) -2.5)
assert (== (at fs 1) 3.75)''',
    'nested': '''let child: array<int> = [7, 8]
let values: array<array<int>> = [child]
let copy: array<array<int>> = (array_slice values 0 1)
let retained: array<int> = (at copy 0)
(array_set child 0 42)
assert (== (at retained 0) 42)
assert (== (array_length copy) 1)''',
    'record': '''let a: Item = Item { value: 7 }
let b: Item = Item { value: 9 }
let values: array<Item> = [a, b]
let copy: array<Item> = (array_slice values 1 1)
let chosen: Item = (at copy 0)
assert (== chosen.value 9)''',
}

class NativeArraySlice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-native-slice-qualified-'))
        print('I retain source, logs and generated artifacts at', cls.work, flush=True)
        cls.sequence = 0

    @classmethod
    def command(cls, *args):
        cls.sequence += 1
        p = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                           text=True, timeout=900)
        (cls.work/f'command-{cls.sequence}.log').write_text(
            repr(tuple(map(str,args)))+'\n'+p.stdout+'\n'+p.stderr)
        if p.returncode:
            raise AssertionError(f'{args}: {p.returncode}\n{p.stdout}\n{p.stderr}')
        return p

    def test_source_and_normal_shadows(self):
        for name, body in CASES.items():
            source = self.work/(name+'.nano')
            source.write_text(('struct Item { value: int }\n' if name=='record' else '')+
                              'fn main() -> int {\n'+body+'\nreturn 0\n}\n'
                              'shadow main { assert (== (main) 0) }\n')
            for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
                with self.subTest(case=name,compiler=compiler):
                    binary = self.work/(name+'-'+compiler)
                    self.command(ROOT/'bin'/compiler,source,'-o',binary)
                    self.command(binary)
            with self.subTest(case=name,compiler='NanoVirt comparison'):
                module = self.work/(name+'.nvm')
                self.command(ROOT/'bin/nano_virt',source,'--emit-nvm','-o',module)
                self.command(ROOT/'bin/nano_vm','--verify-only',module)
                self.command(ROOT/'bin/nano_vm',module)

    def test_arguments_are_evaluated_once(self):
        source = self.work/'once.nano'
        source.write_text("""let mut arrays: int = 0
let mut starts: int = 0
let mut lengths: int = 0
fn values() -> array<int> { set arrays (+ arrays 1) return [7, 8, 9] }
shadow values { set arrays 0 assert (== (array_length (values)) 3) assert (== arrays 1) }
fn start() -> int { set starts (+ starts 1) return 1 }
shadow start { set starts 0 assert (== (start) 1) assert (== starts 1) }
fn length() -> int { set lengths (+ lengths 1) return 2 }
shadow length { set lengths 0 assert (== (length) 2) assert (== lengths 1) }
fn main() -> int {
 set arrays 0 set starts 0 set lengths 0
 let copy: array<int> = (array_slice (values) (start) (length))
 assert (== arrays 1) assert (== starts 1) assert (== lengths 1)
 assert (== (array_length copy) 2) assert (== (at copy 0) 8)
 return 0
}
shadow main { assert (== (main) 0) }
""")
        for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
            with self.subTest(compiler=compiler):
                binary = self.work/('once-'+compiler)
                self.command(ROOT/'bin'/compiler,source,'-o',binary)
                self.command(binary)

    def test_generated_helper_all_kinds_and_bits(self):
        source = (ROOT/'src_nano/transpiler.nano').read_text()
        literals = [json.loads(m.group(1)) for m in re.finditer(
            r'set sb \(cg_append sb ("(?:[^"\\]|\\.)*")\)',source)]
        text = ''.join(literals)
        start = text.index('static DynArray* nl_array_slice(')
        end = text.index('\n}\n',start)+3
        helper = text[start:end]
        harness = (ROOT/'tests/native_array_slice_helper.c').read_text().replace('/* GENERATED_HELPER */',helper)
        c = self.work/'helper.c';c.write_text(harness)
        for cc in ('gcc',os.environ.get('NMS_NATIVE_CLANG','/tmp/nanolang-projection-clang')):
            with self.subTest(compiler=cc):
                binary = self.work/('helper-'+Path(cc).name)
                self.command(cc,'-std=c11','-O2','-Wall','-Wextra','-Werror',
                             '-fsanitize=address,undefined','-fno-sanitize-recover=all',
                             '-I',ROOT/'src/runtime',c,ROOT/'src/runtime/dyn_array.c',
                             ROOT/'src/runtime/gc.c',ROOT/'src/runtime/gc_struct.c','-lm','-o',binary)
                self.command(binary)

if __name__ == '__main__':
    unittest.main()
