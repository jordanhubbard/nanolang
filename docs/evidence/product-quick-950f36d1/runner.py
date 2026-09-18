import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-product-docs-integration')
out=pathlib.Path('/tmp/nanolang-product-quick-950f36d1');out.mkdir(exist_ok=False)
expected='950f36d1e0fed5c7f626b4893566040c45167572'
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert git('rev-parse','HEAD')==expected
assert not git('status','--porcelain','--untracked-files=no')
assert not (root/'bin').exists() and not (root/'obj').exists()
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def artifacts():
 paths=list((root/'bin').glob('*'))
 paths+=list((root/'obj/module_cache').rglob('*.so'))+list((root/'obj/module_cache').rglob('*.dylib'))+list((root/'obj/module_cache').rglob('*.a'))
 return {str(p.relative_to(root)):{'resolved':str(p.resolve()),'sha256':sha(p)} for p in paths if p.is_file()}
def selection():
 return {str(p.relative_to(root)):{'resolved':str(p.resolve()),'sha256':sha(p)} for p in (root/'bin').glob('nanoc*') if p.is_file()}
before=sources();(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n')
env=os.environ.copy()
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):
 env.pop(key,None)
env.update(NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'))
report={'source':expected,'runner_sha256':sha(pathlib.Path(__file__)),'environment':{k:env[k] for k in ('NANO_BUILD_CACHE','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH')},'steps':[],'selection':{},'initial_bin_absent':True,'initial_obj_absent':True}
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
status=1
try:
 for name,cmd in [('bootstrap',['make','-j2','bootstrap']),('tools',['make','-j2','nanoisa_emit','nano_virt','nano_vm','nvm2c','nvm2c-runtime','nanoisa_dump']),('test-quick',['make','test-quick'])]:
  assert git('rev-parse','HEAD')==expected and not git('status','--porcelain','--untracked-files=no')
  if name!='bootstrap':assert (root/'bin/nanoc').resolve()==root/'bin/nanoc_stage2'
  report['selection'][name]=selection()
  (out/(name+'-artifacts-before.json')).write_text(json.dumps(artifacts(),indent=2)+'\n')
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as log:
   result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  status=result.returncode
  report['steps'].append({'name':name,'command':cmd,'status':status,'seconds':round(time.monotonic()-start,3),'sha256':sha(out/(name+'.log'))})
  (out/(name+'-artifacts-after.json')).write_text(json.dumps(artifacts(),indent=2)+'\n')
  (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
  print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=sources();(out/'source-after.json').write_text(json.dumps(after,indent=2)+'\n')
 report['source_unchanged']=before==after
 report['unchanged_head']=git('rev-parse','HEAD')==expected
 report['tracked_tree_clean']=not git('status','--porcelain','--untracked-files=no')
 report['final_selection']=selection()
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'status').write_text(str(status)+'\n')
raise SystemExit(status)
