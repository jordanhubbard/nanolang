import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/private/tmp/nanolang-retirement-20261008');out=Path('/private/tmp/nanolang-retirement-bootstrap-evidence-20261008');out.mkdir()
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD');assert pin.startswith('b76b33a53');assert not git('status','--porcelain')
env={**os.environ,'PATH':'/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:'+os.environ['PATH'],'CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_CC':'/opt/homebrew/opt/llvm/bin/clang','NANOLANG_ROOT':str(root),'NANO_BUILD_CACHE':str(root/'obj/module_cache')}
for name in ('NANOLANG_SELFHOST_COMPILER','NANO_NVM2C','NANOLANG_BOOTSTRAP_NO_CC','NANOLANG_TEST_NVM2C','NANOLANG_PHASE_CSEED'):env.pop(name,None)
command=['make','-j2','bootstrap','CC=/opt/homebrew/opt/llvm/bin/clang']
manifest={'source_commit':pin,'root':str(root),'command':command,'initial_status':'clean','sparse_checkout':git('sparse-checkout','list')}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'runner.py').write_text(Path(__file__).read_text())
print('START raw bootstrap',flush=True);start=time.monotonic()
with (out/'gate.log').open('wb') as log:r=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
manifest.update(exit_code=r.returncode,seconds=time.monotonic()-start,final_status=git('status','--porcelain'),head_unchanged=git('rev-parse','HEAD')==pin,log_sha256=sha(out/'gate.log'))
for name in ('nanoc_seed.nvm','nanoc_stage1.nvm','nanoc_stage2.nvm','nanoc_bootstrap.json'):
 path=root/'bin'/name
 if path.exists():manifest[name]={'bytes':path.stat().st_size,'sha256':sha(path)}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('END',r.returncode,flush=True);raise SystemExit(r.returncode)
