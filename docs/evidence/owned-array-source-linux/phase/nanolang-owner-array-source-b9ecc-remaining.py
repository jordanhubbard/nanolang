import hashlib,importlib.util,json,os,pathlib,shutil,subprocess,sys,time,unittest
root=pathlib.Path('/home/jkh/Src/nanolang-owned-array-source-integrated');fixture_tree=pathlib.Path('/home/jkh/Src/nanolang-owned-array-source-refusal-phase')
out=pathlib.Path('/tmp/nanolang-owner-array-source-b9ecc-remaining');out.mkdir(exist_ok=False);work=out/'work';work.mkdir()
os.chdir(root);sys.path.insert(0,str(root))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def write(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
assert git('rev-parse','HEAD')=='c9fb07ed55329f59e6b67ef0bf3f9dc756d6aa88'
prior=pathlib.Path('/tmp/nanolang-owner-array-source-c9fb-gates');producers=json.loads((prior/'producers-after-setup.json').read_text())
assert all(sha(pathlib.Path(p))==v for p,v in producers.items())
for k,v in json.loads(pathlib.Path('/tmp/nanolang-owner-array-source-c9fb-bootstrap/environment.json').read_text()).items():os.environ[k]=v
os.environ['CC']='/usr/bin/gcc'
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','GNUmakefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def inputs():return {str(p.relative_to(root)):sha(p) for d in ('bin','obj','lib') for p in (root/d).rglob('*') if p.is_file()}
def tools():
 result={}
 for n in ('cc','gcc','clang','make','python3','ar','ld'):
  p=pathlib.Path(shutil.which(n)).resolve();result[n]={'path':str(p),'sha256':sha(p)}
 return result
fixture=fixture_tree/'tests/test_source_owned_float_arrays.py';write('fixture.json',{'path':str(fixture),'sha256':sha(fixture),'pin':subprocess.check_output(['git','rev-parse','HEAD'],cwd=fixture_tree,text=True).strip()})
(out/'test_source_owned_float_arrays.py').write_bytes(fixture.read_bytes());(out/pathlib.Path(__file__).name).write_bytes(pathlib.Path(__file__).read_bytes())
spec=importlib.util.spec_from_file_location('corrected_owned_array_source',fixture);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);cls=module.SourceOwnedFloatArrays
saved=pathlib.Path(json.loads((prior/'setup.json').read_text())['work'])
@classmethod
def setup(c):
 c.work=work;c.emitters=[root/'bin/nanoisa_emit',saved/'nanoc_stage1-emit',saved/'nanoc_stage2-emit'];c.shadow_tools=[saved/(n+'-shadows') for n in ('nanoc_c','nanoc_stage1','nanoc_stage2')]
cls.setUpClass=setup;cls.tearDownClass=classmethod(lambda c:None)
class Result(unittest.TextTestResult):
 def stopTest(self,test):
  super().stopTest(test);dest=out/('artifacts-'+test._testMethodName);shutil.copytree(work,dest);write('artifacts-'+test._testMethodName+'.json',{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})
for n,v in [('source-before.json',sources()),('inputs-before.json',inputs()),('tools-before.json',tools()),('producers-before.json',producers)]:write(n,v)
start=time.monotonic();result=None
try:
 result=unittest.TextTestRunner(verbosity=2,failfast=True,resultclass=Result).run(unittest.TestSuite(cls(n) for n in ('test_exact_type_lexical_and_shadow_refusals','test_unchanged_samples_string_scalar_and_borrow_profiles')))
finally:
 for n,v in [('source-after.json',sources()),('inputs-after.json',inputs()),('tools-after.json',tools()),('producers-after.json',{p:sha(pathlib.Path(p)) for p in producers})]:write(n,v)
 write('status.json',{'seconds':time.monotonic()-start,'testsRun':result.testsRun if result else 0,'errors':len(result.errors) if result else None,'failures':len(result.failures) if result else None,'success':result.wasSuccessful() if result else False})
raise SystemExit(0 if result and result.wasSuccessful() else 1)
