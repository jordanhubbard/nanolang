"""I test copied ordinary-array declarations without runtime admission."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_file_cyclic
ROOT=Path(__file__).resolve().parents[1]
class OrdinaryArrayAuthority(unittest.TestCase):
    command=test_file_cyclic.FileCyclic.command
    @classmethod
    def setUpClass(cls):
        cls.artifacts=Path(tempfile.mkdtemp(prefix='nano-ordinary-array-authority-'))
        print(f'I retain ordinary array authority artifacts at {cls.artifacts}',flush=True)
        cls.cc=shlex.split(os.environ.get('ORDINARY_ARRAY_CC','cc'))
        cls.flags=['-std=c11','-D_DEFAULT_SOURCE','-g','-O1','-Wall','-Wextra','-Werror',*shlex.split(os.environ.get('ORDINARY_ARRAY_CFLAGS',''))]
        cls.objects=shlex.split(os.environ['ORDINARY_ARRAY_OBJECTS'])
        cls.ldflags=shlex.split(os.environ.get('ORDINARY_ARRAY_LDFLAGS','-lm -lcrypto'))
    def test_owned_declarations_and_failures(self):
        for observed in (False,True):
            mode='instrumented' if observed else 'linked';exe=self.artifacts/mode
            sources=['tests/nanoisa/test_ordinary_array_authority.c']
            if not observed:sources+=['src/nanoisa/ownership_contracts.c','src/nanoisa/nvm_v2_layouts.c']
            self.command(mode+'-build',[*self.cc,*self.flags,*(['-DOAA_INSTRUMENT'] if observed else []),*sources,*self.objects,*self.ldflags,'-o',str(exe)])
            output=self.command(mode+'-run',[str(exe)])
            self.assertIn(b'ordinary array declaration checks; no execution authority',output)
            print(output.decode().strip(),flush=True)
if __name__=='__main__':unittest.main()
