import tempfile,sys,unittest
from pathlib import Path
sys.path.insert(0,'/home/jkh/Src/nanolang-record-first-tuple-qualified')
class Retained(tempfile.TemporaryDirectory):
 def __init__(self,*a,**k):
  super().__init__(*a,**k);self._finalizer.detach();print("RETAINED",self.name,flush=True)
 def cleanup(self):pass
tempfile.TemporaryDirectory=Retained
from tests.test_source_borrow_emission import SourceBorrowEmission
suite=unittest.TestSuite([SourceBorrowEmission("test_inline_owner_wrappers_refuse_without_ordinary_fallback")])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
