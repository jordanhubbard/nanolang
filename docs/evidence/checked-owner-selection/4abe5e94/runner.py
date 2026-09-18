import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-checked-owner-dependency-selection')
out=pathlib.Path('/tmp/nanolang-checked-owner-selection-4abe5e94');out.mkdir(exist_ok=False)
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD');assert pin.startswith('4abe5e94');assert not git('status','--porcelain','--untracked-files=no')
assert (root/'bin/nanoc_stage2').exists()
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def tools():return {str(p.relative_to(root)):sha(p) for p in (root/'bin').glob('*') if p.is_file()}
before=sources();(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n');(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
env=os.environ.copy()
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(key,None)
env.update(NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'))
(out/'tools-before.json').write_text(json.dumps(tools(),indent=2)+'\n')
prior=json.loads(pathlib.Path('/tmp/nanolang-checked-owner-selection-32b91d52/bootstrap-tools.json').read_text())
assert all(tools()[p]==prior[p] for p in ('bin/nanoc_c','bin/nanoc_stage1','bin/nanoc_stage2'))
report={'source':pin,'runner_sha256':sha(pathlib.Path(__file__)),'steps':[]};status=1
steps=[('affected-method',['python3','-m','unittest','-v','tests.test_source_borrow_emission.SourceBorrowEmission.test_owned_local_routing_and_shadow_only_admission']),('owned-graphs',['make','test-owned-value-graphs'])]
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
