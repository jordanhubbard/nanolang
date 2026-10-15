import pathlib,subprocess,sys,unittest,tempfile,json,time,hashlib,os
root=pathlib.Path('/private/tmp/nanolang-retirement-20261008')
out=pathlib.Path('/private/tmp/nanolang-snapshot-failed-cases-20261009');out.mkdir(exist_ok=True)
os.chdir(root);sys.path.insert(0,str(root))
def state():
 return {'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'status':subprocess.check_output(['git','status','--porcelain'],text=True), 'sha256':{p:(hashlib.sha256((root/p).read_bytes()).hexdigest() if (root/p).exists() else None) for p in ['src/module_builder.c','tests/test_source_snapshots.py','tests/test_module_cache_publication.py','obj/test_module_generation_probe','bin/nano_as_capture.so']}}
before=state();(out/'before.json').write_text(json.dumps(before,indent=2));assert not before['status']
real_run=subprocess.run
sequence=0
def run(*args,**kwargs):
 global sequence
 sequence+=1;n=sequence;start=time.monotonic();result=None;error=None
 try:
  result=real_run(*args,**kwargs);return result
 except BaseException as e:
  error=e;raise
 finally:
  meta={'argv':list(map(str,args[0])) if args else str(kwargs.get('args')), 'timeout':kwargs.get('timeout'),'seconds':time.monotonic()-start,'exit':getattr(result,'returncode',None),'error':repr(error) if error else None}
  for label in ['stdout','stderr']:
   value=getattr(result if result is not None else error,label,None)
   if value is not None:
    data=value.encode() if isinstance(value,str) else value
    (out/f'{n:04d}-{label}.log').write_bytes(data)
  (out/f'{n:04d}-command.json').write_text(json.dumps(meta,indent=2)+'\n')
class RetainedTemporaryDirectory(tempfile.TemporaryDirectory):
 def cleanup(self):
  self._finalizer.detach()
  with (out/'retained-directories.txt').open('a') as f:f.write(self.name+'\n')
tempfile.TemporaryDirectory=RetainedTemporaryDirectory
subprocess.run=run
from tests.test_source_snapshots import SourceSnapshots
class FailedCases(SourceSnapshots):
 def runTest(self):
  self.assembler_search_order_phases_and_recovery(('xassembler-split',),suffix='.S',cases=[('xassembler-split','package',True,False)])
  self.assembler_search_order_phases_and_recovery(('wa-paired',),suffix='.s',cases=[('wa-paired','common',False,False),('wa-paired','common',True,True),('wa-paired','platform',True,True),('wa-paired','package',False,False),('wa-paired','package',False,True),('wa-paired','package',True,False)])
  self.assembler_search_order_phases_and_recovery(('wa-paired','wa-joined'),suffix='.s',shared_unit=True,cases=[('wa-paired','package',True,False),('wa-joined','common',False,True)])
start=time.monotonic()
try: result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([FailedCases()]))
finally:
 subprocess.run=real_run
 after=state();(out/'after.json').write_text(json.dumps({'state':after,'seconds':time.monotonic()-start,'success':result.wasSuccessful() if 'result' in globals() else False},indent=2))
 assert before==after
sys.exit(not result.wasSuccessful())
