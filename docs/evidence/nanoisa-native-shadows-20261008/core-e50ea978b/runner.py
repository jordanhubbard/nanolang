import subprocess,os,time,json,hashlib,shutil
from pathlib import Path
root=Path('/Users/jordanh/Src/nanolang')
out=Path('/private/tmp/nanolang-core-e50-evidence-20261008');out.mkdir()
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert git('rev-parse','HEAD').startswith('e50ea978b')
assert not git('diff','HEAD','--name-only')
artifact=root/'tests/nanoisa/test_nanoisa_src_nano'
assert not git('ls-files',str(artifact))
artifact.unlink(missing_ok=True)
user=root/'tests/user_guide/refresh_language_pure_function.nano'
before={'head':git('rev-parse','HEAD'),'status':git('status','--porcelain'),'user_sha':sha(user)}
env=dict(os.environ);env['PATH']='/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:'+env['PATH'];env['CC']='/opt/homebrew/opt/llvm/bin/clang'
command=['make','-j2','test-nanoisa-src-nano']
shutil.copy2(__file__,out/'runner.py')
manifest={'before':before,'command':command,'PATH':env['PATH'],'CC':env['CC']}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
start=time.monotonic()
with (out/'gate.log').open('wb') as f:r=subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
manifest.update(exit_code=r.returncode,seconds=time.monotonic()-start,head=git('rev-parse','HEAD'),status=git('status','--porcelain'),tracked_diff=git('diff','HEAD','--name-only'),user_sha=sha(user),log_sha=sha(out/'gate.log'))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('END',r.returncode,flush=True)
raise SystemExit(r.returncode)
