"""I retain generated shadow names through canonical assembly."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class ShadowIdentifiers(unittest.TestCase):
    def command(self,*args):
        p=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr);return p.stdout
    def roundtrip(self,work,text):
        prior=None
        for i in range(3):
            source=work/f'cycle{i}.nasm';module=work/f'cycle{i}.nvm';source.write_text(text)
            self.command(ROOT/'bin/nanoisa','asm',source,'-o',module)
            self.command(ROOT/'bin/nano_vm','--verify-only',module)
            self.command(ROOT/'bin/nano_vm',module)
            dumped=self.command(ROOT/'bin/nanoisa','dump',module)
            if prior is not None:
                self.assertEqual(dumped,text)
                self.assertEqual(module.read_bytes(),prior)
            text=dumped;prior=module.read_bytes()
        return text
    def test_symbolic_generated_names_signatures_calls_and_entry(self):
        source='''.entry $shadow_entry
.function $shadow_entry 0 0 0 int 1
PUSH_I64 41
CALL $shadow_0_length
PUSH_I64 42
EQ
ASSERT
JMP $done
$done:
PUSH_I64 0
RET
.end
.function $shadow_0_length 1 1 0 int 1
.parameters $shadow_0_length int
LOAD_LOCAL 0
PUSH_I64 1
I64_ADD
RET
.end
'''
        with tempfile.TemporaryDirectory(prefix='nano-shadow-symbols-') as temp:
            final=self.roundtrip(Path(temp),source)
            self.assertIn('.function $shadow_entry',final)
            self.assertIn('.function $shadow_0_length',final)
            self.assertIn('.parameters 1 int',final)
    def test_actual_c_shadow_selection_and_names(self):
        with tempfile.TemporaryDirectory(prefix='nano-c-shadows-') as temp:
            work=Path(temp);source=work/'source.nano'
            source.write_text('''fn add_one(value: int) -> int { return (+ value 1) }
shadow add_one { assert (== (add_one 41) 42) }
fn main() -> int { return 0 }
shadow main { assert true }
''')
            text=self.command(ROOT/'obj/borrow_shadow_names',source)
            final=self.roundtrip(work,text)
            self.assertIn('$shadow_0_add_one',final)
            self.assertIn('$shadow_1_main',final)
if __name__=='__main__':unittest.main()
