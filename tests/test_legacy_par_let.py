"""I preserve legacy independent binding groups in both frontends."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
DRIVERS=[Path(p) for p in os.environ.get('NANOLANG_LEGACY_COMPILERS',str(ROOT/'bin/nano_virt')+os.pathsep+str(ROOT/'bin/nanoc_stage2')).split(os.pathsep)]
class LegacyParLet(unittest.TestCase):
    def test_dependencies_and_malformed_groups_preserve_output(self):
        cases=['par-let left=right right=8 in (+ left right)',
               'par-let left=7 right=left in (+ left right)',
               'par-let left=[right] right=8 in (array_length left)',
               'par-let left=7 right= in left',
               'par-let left=7 right=8 left',
               'par-let left=7 in',
               'par-let left=7 in left set left 8',
               'if true {par-let left=7 in left} assert (== left 7)']
        for driver in DRIVERS:
            for group in cases:
                with self.subTest(driver=driver,group=group), tempfile.TemporaryDirectory() as name:
                    path=Path(name);source=path/'case.nano';output=path/'case.nvm'
                    source.write_text('fn main()->int {let right:int=99 '+group+' return 0}\nshadow main {assert true}\n')
                    output.write_bytes(b'previous artifact')
                    command=[driver,source,'-o',output]
                    if driver.name!='nanoc_c': command.append('--emit-nvm')
                    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertEqual(output.read_bytes(),b'previous artifact')
if __name__=='__main__':unittest.main()
