import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/private/tmp/nanolang-bootstrap-65895d5fb')
out=Path('/private/tmp/nanolang-bootstrap-65895d5fb-evidence');out.mkdir()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
assert commit.startswith('65895d5fb')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
env={**os.environ,'PATH':'/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:'+os.environ['PATH'],'CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_NATIVE_TEST_CC':'/opt/homebrew/opt/llvm/bin/clang','NANOLANG_ROOT':str(root),'NANO_MODULE_PATH':str(root/'modules'),'NANO_BUILD_CACHE':str(root/'obj/module_cache')}
for name in ('NANOLANG_SELFHOST_COMPILER','NANO_NVM2C','NANOLANG_BOOTSTRAP_NO_CC'):env.pop(name,None)
manifest={'source_commit':commit,'root':str(root),'steps':[],'initial_status':'clean'}
steps=[('stage1',['make','-j2','bootstrap1']),('stage2',['make','-j2','bootstrap2']),('stage3',['make','-j2','bootstrap3']),('native-verification',['make','verify-bootstrap']),('cli',['make','test-selfhost-cli']),('publication',['make','test-native-product-pipeline','NANO_PIPELINE_PRODUCER=nanoc_stage2'])]
status=1
for name,command in steps:
 print('START',name,flush=True);start=time.monotonic()
 with (out/(name+'.log')).open('wb') as log:
  result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 status=result.returncode
 entry={'name':name,'command':command,'exit_code':status,'seconds':time.monotonic()-start,'log_sha256':hashlib.sha256((out/(name+'.log')).read_bytes()).hexdigest()}
 manifest['steps'].append(entry)
 manifest['final_status']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
 manifest['head_unchanged']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==commit
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('END',name,status,flush=True)
 if status:break
raise SystemExit(status)
