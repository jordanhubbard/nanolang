import os,sys,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,os.getcwd())
from tests.test_owned_global_source import OwnedGlobalSource
original_run=subprocess.run
failures=0
for compiler in ('nanoc_stage1','nanoc_stage2'):
 def run(argv,*args,**kwargs):
  if str(argv[0]).endswith('/bin/nano_virt'):
   argv=[str(Path.cwd()/'bin'/compiler),*argv[1:]]
  return original_run(argv,*args,**kwargs)
 current=type('Global_'+compiler,(OwnedGlobalSource,),{})
 with patch('tests.test_owned_union_c_source.subprocess.run',side_effect=run):
  result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(current))
  failures+=len(result.errors)+len(result.failures)
raise SystemExit(bool(failures))
