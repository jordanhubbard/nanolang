"""I qualify source slicing and native boundaries through actual NanoISA output."""
from pathlib import Path
import os
import shlex
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("NANOLANG_SLICE_BIN", ROOT / "bin"))
NVM2C = Path(os.environ.get("NANOLANG_SLICE_NVM2C", BIN / "nvm2c"))
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
    def command(cls, *args, expected=0, env=None):
        cls.sequence += 1
        p = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                           text=True, timeout=900, env=env)
        (cls.work/f'command-{cls.sequence}.log').write_text(
            repr(tuple(map(str,args)))+'\n'+repr({'ASAN_OPTIONS': (env or os.environ).get('ASAN_OPTIONS')})+'\n'+p.stdout+'\n'+p.stderr)
        if p.returncode != expected:
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
                    self.command(BIN/compiler,source,'-o',binary)
                    self.command(binary)
            with self.subTest(case=name,compiler='NanoVirt comparison'):
                module = self.work/(name+'.nvm')
                self.command(BIN/'nano_virt',source,'--emit-nvm','-o',module)
                self.command(BIN/'nano_vm','--verify-only',module)
                self.command(BIN/'nano_vm',module)

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
                self.command(BIN/compiler,source,'-o',binary)
                self.command(binary)

    def native_compilers(self):
        return [('gcc', shlex.split(os.environ.get('NMS_NATIVE_GCC', 'gcc'))),
                ('clang', shlex.split(os.environ.get('NMS_NATIVE_CLANG', os.environ.get('CC', 'clang'))))]

    def native_environment(self, name):
        env = dict(os.environ)
        override = os.environ.get('NMS_' + name.upper() + '_ASAN_OPTIONS')
        if override is not None:
            env['ASAN_OPTIONS'] = override
        return env

    def test_generated_helper_all_kinds_and_bits(self):
        source = ROOT / 'tests/nanoisa/fixtures/array_slice_all_kinds.nano.txt'
        for compiler in ('nano_virt', 'nanoc_stage1', 'nanoc_stage2'):
            with self.subTest(producer=compiler):
                # I retain a .nano suffix for both installed compiler drivers.
                program = self.work / 'all-kinds.nano'
                program.write_text(source.read_text())
                module = self.work / ('all-kinds-' + compiler + '.nvm')
                self.command(BIN/compiler, program, '--emit-nvm', '-o', module)
                self.command(BIN/'nano_vm', '--verify-only', module)
                self.command(BIN/'nano_vm', module)
                generated = module.with_suffix('.c')
                self.command(NVM2C, module, '-o', generated)
                for name, cc in self.native_compilers():
                    with self.subTest(native=name):
                        binary = self.work / ('all-kinds-' + compiler + '-' + name)
                        self.command(*cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                                     '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                                     generated, '-lm', '-o', binary)
                        self.command(binary, env=self.native_environment(name))

    def test_generated_native_null_boundary(self):
        source = self.work / 'null-boundary.nano'
        source.write_text('fn main() -> int { let values:array<int> = [1] let copy:array<int> = (array_slice values 0 1) return (- (at copy 0) 1) }\nshadow main { assert (== (main) 0) }\n')
        module = source.with_suffix('.nvm')
        self.command(BIN/'nano_virt', source, '--emit-nvm', '-o', module)
        self.command(BIN/'nano_vm', '--verify-only', module)
        self.command(BIN/'nano_vm', module)
        generated = source.with_suffix('.c')
        self.command(NVM2C, module, '-o', generated)
        original = generated.read_text()
        self.assertIn('narr_slice(', original)
        # I call the actual generated native boundary; no helper text is extracted.
        generated.write_text('#define main slice_fixture_main\n' + original +
                             '\n#undef main\nint main(void) { narr_t result = narr_slice(NULL, 1, 2); return (int)result->len; }\n')
        for name, cc in self.native_compilers():
            with self.subTest(native=name):
                binary = self.work / ('null-boundary-' + name)
                self.command(*cc, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                             '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                             generated, '-lm', '-o', binary)
                result = self.command(binary, expected=-signal.SIGABRT, env=self.native_environment(name))
                self.assertIn('I stopped at a native invariant', result.stderr)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertNotIn('ERROR: AddressSanitizer', result.stderr)

if __name__ == '__main__':
    unittest.main()
