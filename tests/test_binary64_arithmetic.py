"""I qualify a helper and its exact standalone source without backend claims."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

class Binary64Arithmetic(unittest.TestCase):
    def setUp(self):
        self.work=Path(tempfile.mkdtemp(prefix='nano-binary64-arithmetic-'))
        self.cc=shlex.split(os.environ.get('CC','cc'))
        self.sequence=0
    def run_command(self,args,success=True):
        self.sequence+=1
        result=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=120)
        (self.work/f'{self.sequence}.log').write_text(shlex.join(list(map(str,args)))+'\n'+result.stdout+result.stderr)
        self.assertEqual(result.returncode==0,success,result.stdout+result.stderr+str(self.work))
        return result
    def test_direct_and_exact_standalone(self):
        self.run_command(['python3',ROOT/'scripts/embed_binary64_arithmetic.py','--check'])
        emitter=self.work/'emit.c'
        emitter.write_text('#include <stdio.h>\n#include "binary64_arithmetic_source.h"\nint main(void){return fputs(nl_binary64_arithmetic_source,stdout)<0;}\n')
        self.run_command(self.cc+['-std=c11','-Wall','-Wextra','-Werror','-I'+str(ROOT/'src'),emitter,'-o',self.work/'emit'])
        header=self.run_command([self.work/'emit']).stdout
        self.assertEqual(header,(ROOT/'src/binary64_arithmetic.h').read_text())
        standalone=self.work/'standalone.c'
        fixture=(ROOT/'tests/test_binary64_arithmetic.c').read_text()
        standalone.write_text(header+'\n'+fixture.replace('#include "binary64_arithmetic.h"',''))
        for source in (ROOT/'tests/test_binary64_arithmetic.c',standalone):
            modes=[['-O0'],['-O2'],['-O3','-ffp-contract=fast'],['-O2','-flto']]
            # I keep Darwin's native linker for its SDK and Apple LTO format.
            if sys.platform != 'darwin' and 'clang' in self.run_command(self.cc+['--version']).stdout.lower():
                modes[-1].append('-fuse-ld=lld')
            for mode in modes:
                exe=self.work/('run'+str(self.sequence))
                self.run_command(self.cc+['-std=c11','-Wall','-Wextra','-Werror','-ffp-contract=off','-fno-fast-math','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(ROOT/'src')]+mode+[source,'-lm','-o',exe])
                self.assertIn('133 arithmetic bit checks',self.run_command([exe]).stdout)
    def test_target_and_fastmath_guards(self):
        text='#include "binary64_arithmetic.h"\nint main(void){return 0;}\n'
        source=self.work/'guard.c';source.write_text(text)
        base=self.cc+['-std=c11','-I'+str(ROOT/'src'),'-c',source,'-o',self.work/'guard.o']
        for flag in ('-ffast-math','-ffinite-math-only'):
            self.assertIn('without fast-math',self.run_command(base+[flag],False).stderr)
        for macro,value,message in [('FLT_EVAL_METHOD','2','evaluated'),('DBL_MANT_DIG','24','binary64 double')]:
            source.write_text(f'#include <float.h>\n#undef {macro}\n#define {macro} {value}\n'+text)
            self.assertIn(message,self.run_command(base,False).stderr)

if __name__=='__main__':unittest.main()
