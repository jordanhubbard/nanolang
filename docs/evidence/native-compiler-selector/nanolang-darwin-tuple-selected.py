from pathlib import Path
import subprocess,os,sys,json,hashlib,time,signal,tempfile,unittest,importlib.util
root=Path('/private/tmp/nanolang-file-cyclic-public-final');report=Path('/tmp/nanolang-darwin-tuple-selected');report.mkdir(exist_ok=False);os.chdir(root);sys.path.insert(0,str(root))
selected=Path('/tmp/nanolang-darwin-aggregate-selectors/tests/test_nanoisa_tuple_values.py');spec=importlib.util.spec_from_file_location('selected_tuple',selected);suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite);suite.ROOT=root
os.environ['NANO_NATIVE_TEST_CC']='/opt/homebrew/opt/llvm/bin/clang';index=0
paths=[selected,Path(__file__),root/'bin/nanoc_c',root/'bin/nano_virt',root/'bin/nano_vm',root/'bin/nanoisa',root/'bin/nvm2c',Path(sys.executable),Path('/opt/homebrew/opt/llvm/bin/clang'),Path('/opt/homebrew/opt/llvm/lib/clang/23/lib/darwin/libclang_rt.asan_osx_dynamic.dylib')]
paths.extend(p for d in ['src','src_nano','modules','stdlib','schema'] for p in (root/d).rglob('*') if p.is_file() and p.suffix in ['.c','.h','.inc','.nano','.json'])
def hashes():return {str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths}
def save(name,d):(report/name).write_text(json.dumps(d,indent=2)+'\n')
def run(args,timeout=120,env=None):
 global index
 label=f'{index:03}';index+=1;argv=list(map(str,args));save(label+'-command.json',{'argv':argv,'cwd':str(root),'timeout':timeout,'selected_ASAN_OPTIONS':(env or os.environ).get('ASAN_OPTIONS')});start=time.monotonic()
 with (report/(label+'.stdout')).open('wb') as out,(report/(label+'.stderr')).open('wb') as err:
  p=subprocess.Popen(argv,cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
  try:rc=p.wait(timeout=timeout);timed=False
  except subprocess.TimeoutExpired:timed=True;os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=10)
 try:os.killpg(p.pid,0);gone=False
 except ProcessLookupError:gone=True
 save(label+'-status.json',{'returncode':rc,'timeout':timed,'seconds':time.monotonic()-start,'leader_reaped':p.poll() is not None,'group_disappeared':gone});assert not timed and gone and rc==0,(label,rc)
 return subprocess.CompletedProcess(argv,rc,(report/(label+'.stdout')).read_text(),(report/(label+'.stderr')).read_text())
save('inputs-before.json',hashes())
emitter=report/'nanoisa_emit';run([root/'bin/nanoc_c','src_nano/nanoisa_emit.nano','-o',emitter,'--keep-c'],900)
fragment=report/'providers.mk';fragment.write_text('root-tuple-providers:\n\t@printf "%s\\n" "$(NANOISA_OBJECTS) $(NANOISA_UTF8)" "$(LDFLAGS)"\n')
selection=run(['make','-s','-f','Makefile.gnu','-f',fragment,'root-tuple-providers']);lines=selection.stdout.splitlines();assert len(lines)==2,lines
import shlex
objects=[root/p for p in shlex.split(lines[0])];providers={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in objects};save('providers-before.json',providers)
comparator=report/'test_nanoisa_src_nano';run(['/usr/bin/cc','-std=c11','-O1','-Wall','-Wextra','-Werror','-Isrc/nanoisa','-Imodules/nanoisa','tests/nanoisa/test_nanoisa_src_nano.c',*objects,*shlex.split(lines[1]),'-o',comparator])
class KeptDirectory:
 def __init__(self,**kwargs):self.name=tempfile.mkdtemp(prefix=kwargs.get('prefix','kept-'),dir=report)
 def __enter__(self):return self.name
 def __exit__(self,*args):return False
suite.tempfile.TemporaryDirectory=KeptDirectory
class Retained(suite.CanonicalTupleValues):
 def checked(self,*args,env=None):
  args=list(args)
  if str(args[0])==str(root/'bin/nanoisa_emit'):args[0]=emitter
  if str(args[0])==str(root/'tests/nanoisa/test_nanoisa_src_nano'):args[0]=comparator
  result=run(args,120,env);self.assertEqual(result.returncode,0,result.stdout+result.stderr);return result
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([Retained('test_both_producers_preserve_tuple_calls_results_and_projection')]))
save('inputs-after.json',hashes());assert json.loads((report/'inputs-before.json').read_text())==hashes();after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in objects};save('providers-after.json',after);assert providers==after
save('products.json',{str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in report.rglob('*') if p.is_file()});save('result.json',{'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful()});sys.exit(0 if result.wasSuccessful() else 1)
