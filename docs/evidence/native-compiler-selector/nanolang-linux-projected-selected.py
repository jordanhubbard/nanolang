from pathlib import Path
import os,sys,json,hashlib,subprocess,signal,time,tempfile,unittest
root=Path('/home/jkh/Src/nanolang-file-cyclic-public-qualified-4c85');report=Path('/tmp/nanolang-linux-projected-selected');report.mkdir(exist_ok=False)
os.chdir(root);sys.path.insert(0,str(root))
import importlib.util
selected=Path('/home/jkh/Src/nanolang-capture-flags/tests/test_nanoisa_flat_records.py')
spec=importlib.util.spec_from_file_location('selected_flat',selected);suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite);suite.ROOT=root
os.environ.pop('NANO_NATIVE_TEST_CC',None)
index=0
paths=[root/'tests/test_nanoisa_flat_records.py',root/'bin/nanoisa',root/'bin/nano_vm',root/'bin/nvm2c',Path('/usr/bin/cc'),Path(sys.executable),Path(__file__),selected,Path('/usr/bin/gcc'),Path(subprocess.check_output(['/usr/bin/gcc','-print-file-name=libasan.so'],text=True).strip())]
def hashes():return {str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths}
def save(name,data):(report/name).write_text(json.dumps(data,indent=2)+'\n')
save('inputs-before.json',hashes());save('environment.json',{k:os.environ.get(k) for k in ['PATH','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS','CC','SDKROOT','DEVELOPER_DIR','NANO_NATIVE_TEST_CC']})
class KeptDirectory:
 def __init__(self,**kwargs):self.name=tempfile.mkdtemp(prefix=kwargs.get('prefix','kept-'),dir=report)
 def __enter__(self):return self.name
 def __exit__(self,*args):return False
suite.tempfile.TemporaryDirectory=KeptDirectory
class Retained(suite.FlatRecordEmitter):
 def run_checked(self,*args):
  global index
  label=f'{index:03}';index+=1;argv=[str(x) for x in args];save(label+'-command.json',{'argv':argv,'cwd':str(root),'timeout':120});start=time.monotonic()
  with (report/(label+'.stdout')).open('wb') as out,(report/(label+'.stderr')).open('wb') as err:
   p=subprocess.Popen(argv,cwd=root,stdout=out,stderr=err,start_new_session=True)
   try:code=p.wait(timeout=120);timeout=False
   except subprocess.TimeoutExpired:
    timeout=True;os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
  try:os.killpg(p.pid,0);gone=False
  except ProcessLookupError:gone=True
  save(label+'-status.json',{'returncode':code,'timeout':timeout,'seconds':time.monotonic()-start,'leader_reaped':p.poll() is not None,'group_disappeared':gone})
  stdout=(report/(label+'.stdout')).read_text();stderr=(report/(label+'.stderr')).read_text()
  self.assertFalse(timeout,'I retained the original120-second timeout');self.assertTrue(gone);self.assertEqual(code,0,stdout+stderr)
  return subprocess.CompletedProcess(argv,code,stdout,stderr)
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([Retained('test_projected_record_array_global_field_executes_in_both_orders')]))
save('inputs-after.json',hashes());assert json.loads((report/'inputs-before.json').read_text())==hashes()
save('products.json',{str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in report.rglob('*') if p.is_file()})
save('result.json',{'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful()});sys.exit(0 if result.wasSuccessful() else 1)
