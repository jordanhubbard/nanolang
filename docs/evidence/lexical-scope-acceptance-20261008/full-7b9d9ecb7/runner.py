import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/Users/jordanh/Src/nanolang');out=Path('/private/tmp/nanolang-product-7b9d9ecb7-evidence-20261008');out.mkdir()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
assert commit == '7b9d9ecb7628136e2abfcd40419f848984225835'
initial_status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
assert initial_status == '?? tests/user_guide/refresh_language_pure_function.nano'
user_file=root/'tests/user_guide/refresh_language_pure_function.nano'
user_sha=hashlib.sha256(user_file.read_bytes()).hexdigest()
env={**os.environ,'PATH':'/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:'+os.environ['PATH'],'CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_CC':'/opt/homebrew/opt/llvm/bin/clang','NANOLANG_ROOT':str(root),'NANO_BUILD_CACHE':str(root/'obj/module_cache')}
for name in ('NANOLANG_SELFHOST_COMPILER','NANO_NVM2C','NANOLANG_BOOTSTRAP_NO_CC','NANOLANG_TEST_NVM2C','NANOLANG_PHASE_CSEED'):env.pop(name,None)
command=['make','-j2','test-one-ir-compiler']
manifest={'source_commit':commit,'root':str(root),'command':command,'initial_status':initial_status,'tracked_tree_initially_clean':True,'preserved_user_file_sha256':user_sha}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'runner.py').write_text(Path(__file__).read_text())
print('START full compiler-product gate',flush=True);start=time.monotonic()
with (out/'gate.log').open('wb') as log:result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
manifest.update(user_file_unchanged=hashlib.sha256(user_file.read_bytes()).hexdigest()==user_sha,exit_code=result.returncode,seconds=time.monotonic()-start,log_sha256=hashlib.sha256((out/'gate.log').read_bytes()).hexdigest(),final_status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),head_unchanged=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==commit)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('END',result.returncode,flush=True)
raise SystemExit(result.returncode)
