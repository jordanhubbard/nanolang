"""I qualify private structural layouts, never executable array authority."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests.test_file_indirect_flow import FileIndirectFlow
ROOT=Path(__file__).resolve().parents[1]
class OwnershipLayoutsPrivate(unittest.TestCase):
    command=FileIndirectFlow.command
    @classmethod
    def setUpClass(cls):
        cls.artifacts=Path(tempfile.mkdtemp(prefix='nano-ownership-layouts-'))
        print(f'I retain private layout artifacts at {cls.artifacts}',flush=True)
        cls.cc=shlex.split(os.environ.get('OWNERSHIP_LAYOUT_CC','cc'))
        cls.flags=['-std=c11','-g','-O1','-Wall','-Wextra','-Werror',*shlex.split(os.environ.get('OWNERSHIP_LAYOUT_CFLAGS',''))]
    def test_structural_profiles_and_failures(self):
        for observed in (False,True):
            mode='instrumented' if observed else 'linked';exe=self.artifacts/mode
            sources=['tests/nanoisa/test_ownership_layouts_private.c','src/nanoisa/nvm_v2_cursor.c']
            if not observed:sources.append('src/nanoisa/nvm_v2_layouts.c')
            self.command(mode+'-build',[*self.cc,*self.flags,*(['-DLAYOUT_INSTRUMENT'] if observed else []),*sources,'-o',str(exe)])
            output=self.command(mode+'-run',[str(exe)])
            self.assertIn(b'private layout structure checks; no array authority',output)
            print(output.decode().strip(),flush=True)
if __name__=='__main__':unittest.main()
