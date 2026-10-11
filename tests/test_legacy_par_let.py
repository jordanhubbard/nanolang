"""I preserve legacy independent binding groups in both frontends."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
DRIVERS=[Path(p) for p in os.environ.get('NANOLANG_LEGACY_COMPILERS',str(ROOT/'bin/nano_virt')+os.pathsep+str(ROOT/'bin/nanoc_stage2')).split(os.pathsep)]
class LegacyParLet(unittest.TestCase):
    def test_source_visibility_and_discarded_native_bodies(self):
        cases = [
            ('fn main()->int {par-let left=(array_length [1]) array_length=7 in left return (+ left array_length)}', 8),
            ('fn main()->int {\npar-let left=(array_length [1])\n array_length=7\n in left\n return (+ left array_length)\n}', 8),
            ('let mut trace:int=0 fn note()->void {set trace 3} shadow note {assert true} fn main()->int {par-let value=7 in (note) return trace}', 3),
            ('fn earlier()->int {return (array_length [1,2])} shadow earlier {assert true} fn main()->int {par-let array_length=7 in 0 return (earlier)}', 2),
        ]
        for body in ['(+ value 2)', '1.5', '"kept"', '[7,9]', '(7,true)', 'Item {value:7}']:
            cases.append(('struct Item {value:int} fn main()->int {par-let value=7 in '+body+' return value}', 7))
        for driver in DRIVERS:
            for source_text, expected in cases:
                with self.subTest(driver=driver,source=source_text), tempfile.TemporaryDirectory() as name:
                    path=Path(name);source=path/'case.nano';output=path/'case.out'
                    source.write_text(source_text+'\nshadow main {assert true}\n')
                    command=[driver,source,'-o',output]
                    if driver.name!='nanoc_c': command.append('--emit-nvm')
                    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertNotIn('TYPE MISMATCH',result.stderr)
                    self.assertNotIn('Error at',result.stderr)
                    run=[output] if driver.name=='nanoc_c' else [ROOT/'bin/nano_vm',output]
                    executed=subprocess.run(run,cwd=ROOT,capture_output=True,text=True,timeout=120)
                    self.assertEqual(executed.returncode,expected,executed.stdout+executed.stderr)

    def test_dependencies_and_malformed_groups_preserve_output(self):
        cases=['par-let left=right right=8 in (+ left right)',
               'par-let left=7 right=left in (+ left right)',
               'par-let left=[right] right=8 in (array_length left)',
               'par-let left=7 right= in left',
               'par-let left=7 right=8 left',
               'par-let left=7 in',
               'par-let left=7 in left set left 8',
               'par-let array_length=7 in (array_length [1])',
               'par-let array_length=7 in 0 return (array_length [1])',
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
