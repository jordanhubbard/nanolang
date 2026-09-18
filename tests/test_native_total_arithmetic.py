"""I preserve total integer semantics in generated C without signed overflow."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class NativeTotalArithmetic(unittest.TestCase):
    def checked(self, args):
        result = subprocess.run(list(map(str,args)), cwd=ROOT, capture_output=True,
                                text=True, timeout=60,
                                env={**os.environ,'UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'})
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertNotIn('runtime error:',result.stderr)
        return result

    def test_typed_and_generic_integer_boundaries(self):
        low, high = -(1 << 63), (1 << 63)-1
        cases = [('ADD',high,1,low),('ADD',low,-1,high),('SUB',low,1,high),
                 ('SUB',high,-1,low),('MUL',high,2,-2),('MUL',low,-1,low),
                 ('MUL',0,low,0),('DIV',low,-1,low),('DIV',low,0,0),
                 ('DIV',-17,5,-3),('DIV',17,-5,-3),('MOD',low,-1,0),
                 ('MOD',low,0,0),('MOD',-17,5,-2),('MOD',17,-5,2)]
        with tempfile.TemporaryDirectory(prefix='nano-total-arithmetic-') as directory:
            work=Path(directory)
            for typed in (False,True):
                body=''
                for op,a,b,expected in cases:
                    name=('I64_'+op+('_S' if op in ('DIV','MOD') else '')) if typed else op
                    if typed and op=='MOD': name='I64_REM_S'
                    body+=f'PUSH_I64 {a}\nPUSH_I64 {b}\n{name}\nDUP\nTYPE_CHECK 1\nASSERT\nPUSH_I64 {expected}\nI64_EQ\nASSERT\n'
                for value,expected in [(low,low),(high,-high),(-1,1),(0,0)]:
                    name='I64_NEG' if typed else 'NEG'
                    body+=f'PUSH_I64 {value}\n{name}\nPUSH_I64 {expected}\nI64_EQ\nASSERT\n'
                assembly=work/'input.nasm';module=work/'input.nvm';source=work/'output.c'
                assembly.write_text('.entry main\n.function main 0 0 0 int 1\n'+body+'PUSH_I64 0\nRET\n.end\n')
                self.checked([ROOT/'bin/nanoisa','asm',assembly,'-o',module])
                self.checked([ROOT/'bin/nano_vm','--verify-only',module])
                self.checked([ROOT/'bin/nano_vm',module])
                self.checked([ROOT/'bin/nvm2c',module,'-o',source])
                for optimize in ('-O0','-O2'):
                    with self.subTest(typed=typed,optimize=optimize):
                        output=work/'program'
                        self.checked([os.environ.get('CC','cc'),'-std=c11',optimize,'-Wall','-Wextra','-Werror',
                                      '-fsanitize=undefined','-fno-sanitize-recover=all',source,'-o',output])
                        self.checked([output])

if __name__=='__main__': unittest.main()
