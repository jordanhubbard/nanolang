import pathlib,shutil,unittest,json,sys,time
from tests import test_managed_binary64_parse as m
out=pathlib.Path(sys.argv[1])
old=m.Binary64Parse.setUp
def setup(self):
 old(self)
 self.addCleanup(lambda:shutil.copytree(self.work,out/'artifacts',dirs_exist_ok=True))
m.Binary64Parse.setUp=setup
oldrun=m.Binary64Parse.run_cmd
seq=0
def run(self,args,*a,**kw):
 global seq
 seq+=1
 p=out/('command-%02d.log'%seq)
 p.write_text(repr([str(x) for x in args])+'\n')
 try:
  result=oldrun(self,args,*a,**kw)
  with p.open('a') as f:f.write(result.stdout+'\n'+result.stderr)
  return result
 except BaseException as e:
  with p.open('a') as f:f.write(repr(e)+'\n')
  raise
m.Binary64Parse.run_cmd=run
suite=unittest.TestSuite([m.Binary64Parse('test_core_bits_and_ownership_match_reference_on_native_and_wasm')])
result=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
