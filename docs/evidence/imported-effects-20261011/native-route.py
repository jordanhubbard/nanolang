import subprocess,unittest
from pathlib import Path
from tests import test_imported_effect_identity as matrix
matrix.DRIVERS=[str(Path.cwd()/'bin/nanoc_c')]
def compile_native(self,driver,source,output):
 return subprocess.run([driver,str(source),'-o',str(output)],capture_output=True,text=True,timeout=120)
def execute_native(self,output,expected):
 r=subprocess.run([str(output)],capture_output=True,text=True,timeout=60)
 self.assertEqual(r.returncode,expected,r.stdout+r.stderr)
matrix.ImportedEffectIdentity.compile=compile_native
matrix.ImportedEffectIdentity.execute_module=execute_native
suite=unittest.defaultTestLoader.loadTestsFromTestCase(matrix.ImportedEffectIdentity)
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
