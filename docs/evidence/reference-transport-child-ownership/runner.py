import os,sys,json,hashlib,subprocess,pathlib,importlib.util,unittest,time,shlex,shutil
root=pathlib.Path.cwd();out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
os.environ['NANO_NATIVE_TEST_CC']=sys.argv[2]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
files=subprocess.check_output(['git','ls-files','src','src_nano','tests','scripts','GNUmakefile','Makefile','Makefile.gnu'],text=True).splitlines()
def inventory():return {p:sha(root/p) for p in files if (root/p).is_file()}
before=inventory();(out/'before.json').write_text(json.dumps(before,sort_keys=True,indent=2))
cc=pathlib.Path(shutil.which(shlex.split(sys.argv[2])[0])).resolve();cc_hash=sha(cc)
(out/'environment.json').write_text(json.dumps({'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'compiler':str(cc),'compiler_sha256':cc_hash,'compiler_version':subprocess.check_output([str(cc),'--version'],text=True),'python':sys.version,'platform':sys.platform,'runner_sha256':sha(__file__)},indent=2))
spec=importlib.util.spec_from_file_location('transport',root/'tests/test_reference_eval_transport.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_teardown=m.ReferenceTransport.tearDownClass
@classmethod
def teardown(cls):
 shutil.copytree(cls.base,out/'artifacts');original_teardown()
m.ReferenceTransport.tearDownClass=teardown
original_run=subprocess.run;commands=[]
def recorded(command,*args,**kwargs):
 result=original_run(command,*args,**kwargs)
 commands.append({'command':[str(x) for x in command],'status':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'asan_options':kwargs.get('env',{}).get('ASAN_OPTIONS')})
 (out/'commands.json').write_text(json.dumps(commands,indent=2));return result
subprocess.run=recorded
start=time.monotonic()
with (out/'suite.log').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2,failfast=True).run(unittest.defaultTestLoader.loadTestsFromTestCase(m.ReferenceTransport))
after=inventory();(out/'after.json').write_text(json.dumps(after,sort_keys=True,indent=2))
summary={'success':result.wasSuccessful(),'tests':result.testsRun,'seconds':time.monotonic()-start,'sources_unchanged':before==after,'compiler_unchanged':cc_hash==sha(cc),'compiler_after_sha256':sha(cc)}
(out/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));sys.exit(0 if all([summary['success'],summary['sources_unchanged'],summary['compiler_unchanged']]) else 1)
