from pathlib import Path
import os,sys,tempfile,unittest
sys.path.insert(0,str(Path.cwd()))
from tests import test_nanoisa_functional_arrays as scalar
from tests import test_native_callable_arrays as arrays
from tests import test_native_functional_callbacks as functional
class Scalar(scalar.FunctionalArrays):
 @classmethod
 def setUpClass(cls):cls.driver=Path('/private/tmp/nl51-selfhost-functional/emitter')
 @classmethod
 def tearDownClass(cls):pass
class Loops(arrays.NativeCallableArrays):
 def execute(self,source):
  with tempfile.TemporaryDirectory() as tmp:
   work=Path(tmp);path=work/'source.nano';path.write_text(arrays.PRELUDE+source+'\nshadow main {assert (== (main) 0)}\n')
   functional.NativeFunctionalCallbacks().execute(path,work)
suite=unittest.TestSuite()
for name in unittest.defaultTestLoader.getTestCaseNames(Scalar):
 if name!='test_bound_callback_owners_and_selected_shadow_dependencies':suite.addTest(Scalar(name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
