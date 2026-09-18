"""I qualify exact source bit transport through fresh compiler producers."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class Binary64SourceTransport(unittest.TestCase):
    def command(self,*args):
        p=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=180)
        self.assertEqual(p.returncode,0,f'{args}\n{p.stdout}\n{p.stderr}')
        return p
    def test_source_producers_and_legacy_c(self):
        source=ROOT/'tests/nanoisa/fixtures/binary64_bits.nano'
        with tempfile.TemporaryDirectory(prefix='binary64-source-') as tmp:
            work=Path(tmp)
            for name in ('nano_virt','nanoc_stage1','nanoc_stage2'):
                with self.subTest(producer=name):
                    module=work/(name+'.nvm')
                    self.command(ROOT/'bin'/name,source,'--emit-nvm','-o',module)
                    self.command(ROOT/'bin/nano_vm','--verify-only',module)
                    self.command(ROOT/'bin/nano_vm',module)
                    c=work/(name+'.c');exe=work/name
                    self.command(ROOT/'bin/nvm2c',module,'-o',c)
                    self.command(os.environ.get('CC','cc'),'-std=c11','-O2','-Wall','-Wextra','-Werror',
                                 '-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'-lm','-o',exe)
                    self.command(exe)
            for name in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
                with self.subTest(legacy=name):
                    exe=work/(name+'-legacy')
                    self.command(ROOT/'bin'/name,source,'-o',exe)
                    self.command(exe)
if __name__=='__main__':unittest.main()
