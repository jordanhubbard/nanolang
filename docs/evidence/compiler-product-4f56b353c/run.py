import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/private/tmp/nanolang-callables-4f56b353c');out=Path('/private/tmp/nanolang-callables-4f56b353c-evidence');out.mkdir()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
assert commit.startswith('4f56b353c')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
env={**os.environ,'PATH':'/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:'+os.environ['PATH'],'CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_CC':'/opt/homebrew/opt/llvm/bin/clang','NANOLANG_ROOT':str(root),'NANO_BUILD_CACHE':str(root/'obj/module_cache')}
for name in ('NANOLANG_SELFHOST_COMPILER','NANO_NVM2C','NANOLANG_BOOTSTRAP_NO_CC'):env.pop(name,None)
command=['make','-j2','test-one-ir-compiler']
manifest={'source_commit':commit,'root':str(root),'command':command,'initial_status':'clean'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('START full compiler-product gate',flush=True);start=time.monotonic()
with (out/'gate.log').open('wb') as log:result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
manifest.update(exit_code=result.returncode,seconds=time.monotonic()-start,log_sha256=hashlib.sha256((out/'gate.log').read_bytes()).hexdigest(),final_status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),head_unchanged=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==commit)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('END',result.returncode,flush=True)
raise SystemExit(result.returncode)
