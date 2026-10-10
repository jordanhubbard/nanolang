"""I retain module identity for private C-seed globals and their initialization."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CSeedGlobalIdentity(unittest.TestCase):
    def check(self, files):
        with tempfile.TemporaryDirectory(prefix='nano-global-identity-') as tmp:
            work = Path(tmp)
            for name, text in files.items(): (work/name).write_text(text)
            module, source, binary = (work/name for name in ('main.nvm','main.c','main'))
            cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC',os.environ.get('CC','cc')))
            commands = [[ROOT/'bin/nano_virt',work/'main.nano','--emit-nvm','-o',module],
                        [ROOT/'bin/nano_vm','--verify-only',module],[ROOT/'bin/nano_vm',module],
                        [ROOT/'bin/nvm2c',module,'-o',source],
                        cc+['-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',source,ROOT/'bin/nano_aot_runtime.o','-lm','-o',binary],[binary]]
            for command in commands:
                result=subprocess.run(list(map(str,command)),cwd=ROOT,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1'},capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_distinct_types_and_root_binding(self):
        self.check({
            'a.nano':'let answer: int = 41\npub fn read_answer()->int{return answer}\nshadow read_answer {assert (== (read_answer) 41)}\n',
            'b.nano':'let answer: string = "second"\npub fn read_answer()->string{return answer}\nshadow read_answer {assert (== (read_answer) "second")}\n',
            'main.nano':'module "a.nano" as first\nmodule "b.nano" as second\nlet answer: int = 100\nfn fn_0()->int{return 13}\nshadow fn_0 {assert (== (fn_0) 13)}\nfn main()->int{assert (== (fn_0) 13) assert (== answer 100) assert (== (first.read_answer) 41) assert (== (second.read_answer) "second") return 0}\nshadow main {assert (== (main) 0)}\n'})

    def test_long_function_names_keep_distinct_native_targets(self):
        first, second = 'x' * 300, 'x' * 299 + 'y'
        source = (f'fn {first}()->int{{return 17}}\nshadow {first} {{assert (== ({first}) 17)}}\n'
                  f'fn {second}()->int{{return 25}}\nshadow {second} {{assert (== ({second}) 25)}}\n'
                  f'fn main()->int{{assert (== (+ ({first}) ({second})) 42) return 0}}\n'
                  'shadow main {assert (== (main) 0)}\n')
        self.check({'main.nano': source})

    def test_initializer_context_and_repeated_import(self):
        self.check({
            'a.nano':'let seed: int = 41\nlet answer: int = (+ seed 1)\npub fn read_answer()->int{return answer}\nshadow read_answer {assert (== (read_answer) 42)}\n',
            'b.nano':'let seed: string = "second"\nlet answer: string = seed\npub fn read_answer()->string{return answer}\nshadow read_answer {assert (== (read_answer) "second")}\n',
            'main.nano':'module "a.nano" as first\nmodule "b.nano" as second\nmodule "a.nano" as again\nfn main()->int{assert (== (first.read_answer) 42) assert (== (again.read_answer) 42) assert (== (second.read_answer) "second") return 0}\nshadow main {assert (== (main) 0)}\n'})


if __name__ == '__main__':
    unittest.main()
