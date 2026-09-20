"""I test complete copied mixed declarations without runtime admission."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import test_file_cyclic
ROOT=Path(__file__).resolve().parents[1]
class OwnershipDeclarationProjection(unittest.TestCase):
    command=test_file_cyclic.FileCyclic.command
    @classmethod
    def setUpClass(cls):
        cls.artifacts=Path(tempfile.mkdtemp(prefix='nano-ownership-declaration-'))
        print(f'I retain mixed declaration artifacts at {cls.artifacts}',flush=True)
        cls.cc=shlex.split(os.environ.get('DECLARATION_CC','cc'))
        cls.flags=['-std=c11','-D_DEFAULT_SOURCE','-g','-O1','-Wall','-Wextra','-Werror',*shlex.split(os.environ.get('DECLARATION_CFLAGS',''))]
        cls.objects=[p for p in shlex.split(os.environ['DECLARATION_OBJECTS'])
                     if Path(p).stem not in ('ownership_contracts','nvm_v2_layouts')]
        cls.ldflags=shlex.split(os.environ.get('DECLARATION_LDFLAGS','-lm -lcrypto'))
    def test_owned_declarations_and_failures(self):
        for observed in (False,True):
            mode='instrumented' if observed else 'linked';exe=self.artifacts/mode
            sources=['tests/nanoisa/test_ownership_declaration_projection.c']
            if not observed:sources+=['src/nanoisa/ownership_contracts.c','src/nanoisa/nvm_v2_layouts.c']
            self.command(mode+'-build',[*self.cc,*self.flags,*(['-DOAA_INSTRUMENT'] if observed else []),*sources,*self.objects,*self.ldflags,'-o',str(exe)])
            output=self.command(mode+'-run',[str(exe)])
            self.assertIn(b'complete mixed declaration checks; no execution authority',output)
            print(output.decode().strip(),flush=True)
if __name__=='__main__':unittest.main()
