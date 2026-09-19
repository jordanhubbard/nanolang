import hashlib,json,os,pathlib,shutil,subprocess,sys,time,unittest
root=pathlib.Path('/home/jkh/Src/nanolang-owned-array-source-integrated');out=pathlib.Path('/tmp/nanolang-owner-array-source-c9fb-gates');out.mkdir(exist_ok=False)
os.chdir(root);sys.path.insert(0,str(root))
pin='c9fb07ed55329f59e6b67ef0bf3f9dc756d6aa88'
def git(*a):return subprocess.check_output(['git',*a],text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert git('rev-parse','HEAD')==pin and not git('status','--porcelain')
assert all(x['status']==0 for x in json.loads(pathlib.Path('/tmp/nanolang-owner-array-source-c9fb-bootstrap/status.json').read_text())['steps'])
for k,v in json.loads(pathlib.Path('/tmp/nanolang-owner-array-source-c9fb-bootstrap/environment.json').read_text()).items():os.environ[k]=v
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','GNUmakefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def inputs():return {str(p.relative_to(root)):sha(p) for d in ('bin','obj','lib') for p in (root/d).rglob('*') if p.is_file()}
def write(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
def tools():
 result={}
 for n in ('cc','gcc','clang','make','python3','ar','ld'):
  p=pathlib.Path(shutil.which(n)).resolve();result[n]={'path':str(p),'sha256':sha(p)}
 return result
write('host-tools-before.json',tools());write('source-before.json',sources());write('inputs-before-setup.json',inputs());(out/pathlib.Path(__file__).name).write_bytes(pathlib.Path(__file__).read_bytes())
from tests import test_source_owned_float_arrays as fixture
cls=fixture.SourceOwnedFloatArrays;original_setup=cls.setUpClass
@classmethod
def setup(c):
 start=time.monotonic()
 try:original_setup()
 finally:
  write('inputs-after-setup.json',inputs());write('setup.json',{'seconds':time.monotonic()-start,'work':str(getattr(c,'work',''))})
 os.environ['CC']='/usr/bin/gcc'
 producers=[root/'bin'/n for n in ('nanoc_c','nanoc_stage1','nanoc_stage2','nano_virt','nano_vm','nvm2c','nanoisa','nanoisa_emit')]+c.emitters+c.shadow_tools
 write('producers-after-setup.json',{str(p):sha(p) for p in producers})
 p=pathlib.Path(shutil.which(os.environ['CC'])).resolve();write('native-compiler.json',{'path':str(p),'sha256':sha(p),'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'})
cls.setUpClass=setup
class Result(unittest.TextTestResult):
 def stopTest(self,test):
  super().stopTest(test)
  if hasattr(cls,'work'):
   dest=out/('artifacts-'+test._testMethodName);shutil.copytree(cls.work,dest)
   write('artifacts-'+test._testMethodName+'.json',{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})
start=time.monotonic();result=None
try:
 result=unittest.TextTestRunner(verbosity=2,failfast=True,resultclass=Result).run(fixture.load_tests(None,None,None))
finally:
 write('host-tools-after.json',tools());write('source-after.json',sources());write('inputs-after-tests.json',inputs());write('status.json',{'pin':pin,'seconds':time.monotonic()-start,'testsRun':result.testsRun if result else 0,'errors':len(result.errors) if result else None,'failures':len(result.failures) if result else None,'success':result.wasSuccessful() if result else False})
raise SystemExit(0 if result and result.wasSuccessful() else 1)
