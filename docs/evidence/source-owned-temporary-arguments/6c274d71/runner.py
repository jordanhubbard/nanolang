import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-source-owned-temporary-arguments')
out=pathlib.Path('/tmp/nanolang-temporary-owner-6c274d71');out.mkdir(exist_ok=False)
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD');assert pin.startswith('6c274d71');assert not git('status','--porcelain','--untracked-files=no')
assert not (root/'bin').exists() and not (root/'obj').exists()
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def tools():return {str(p.relative_to(root)):sha(p) for p in (root/'bin').glob('*') if p.is_file()}
before=sources();(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n');(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
env=os.environ.copy()
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(key,None)
env.update(NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'))
report={'source':pin,'runner_sha256':sha(pathlib.Path(__file__)),'steps':[]};status=1
methods=['test_scalar_terminal_operation','test_empty_resource_pattern','test_initializer_evaluated_once','test_existing_resource_examples']
# Existing-resource examples includes the unchanged broader fixture; its first refusal is retained.
steps=[('bootstrap',['make','-j2','bootstrap']),('paired-source',['make','test-source-borrow-emission']),('original-patterns',['python3','-m','unittest','-v',*[f'tests.test_owned_record_patterns.OwnedRecordPatterns.{m}' for m in methods]]),('runtime',['make','test-owned-value-graphs','test-owned-value-results','test-consuming-calls','test-multiple-consuming-calls','test-owned-assertions'])]
steps[2],steps[3]=steps[3],steps[2]
try:
 for name,cmd in steps:
  assert git('rev-parse','HEAD')==pin and not git('status','--porcelain','--untracked-files=no')
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as log:result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  status=result.returncode;report['steps'].append({'name':name,'command':cmd,'status':status,'seconds':round(time.monotonic()-start,3),'sha256':sha(out/(name+'.log'))})
  (out/(name+'-tools.json')).write_text(json.dumps(tools(),indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=sources();(out/'source-after.json').write_text(json.dumps(after,indent=2)+'\n');report['source_unchanged']=after==before;report['unchanged_head']=git('rev-parse','HEAD')==pin;report['tracked_tree_clean']=not git('status','--porcelain','--untracked-files=no');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');(out/'status').write_text(str(status)+'\n')
raise SystemExit(status)
