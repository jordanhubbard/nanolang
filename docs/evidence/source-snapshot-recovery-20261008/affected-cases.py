import sys,unittest
sys.path.insert(0, '/Users/jordanh/Src/nanolang')
from tests.test_source_snapshots import SourceSnapshots
class Affected(SourceSnapshots):
 def raw(self): self.run_affected('.s',False)
 def preprocessed(self): self.run_affected('.S',False)
 def shared_raw(self): self.run_affected('.s',True)
 def shared_preprocessed(self): self.run_affected('.S',True)
 def run_affected(self,suffix,shared_unit):
  self.assembler_search_order_phases_and_recovery(('xassembler-split',),suffix=suffix,shared_unit=shared_unit,
   cases=tuple(('xassembler-split','common',external,shared) for external in (False,True) for shared in (False,True)))
r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Affected(n) for n in ('raw','preprocessed','shared_raw','shared_preprocessed')))
raise SystemExit(not r.wasSuccessful())
