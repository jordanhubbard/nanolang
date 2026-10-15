import hashlib,json,os,subprocess,time,shutil
from pathlib import Path
root=Path('/private/tmp/nanolang-retirement-20261008')
out=Path('/private/tmp/nanolang-snapshot-shell-corrected-20261008');out.mkdir()
def git(*args): return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD')
assert pin=='fdca4ee9b53b816232074b0d539893168715baf2'
assert not git('status','--porcelain')
files=git('ls-files','src','tests','scripts','Makefile','Makefile.gnu').splitlines()
def snapshot(): return {p:sha(root/p) for p in files if (root/p).is_file()}
before=snapshot()
(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n')
shutil.copy2(__file__,out/'runner.py')
env=dict(os.environ)
for name in ('NANO_CC','CC','NANOLANG_ROOT','NANO_BUILD_CACHE','NANO_CAPTURE_TIMEOUT_MS'):env.pop(name,None)
env['PATH']='/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin'
command=['/opt/homebrew/bin/python3','-m','unittest','-v','tests.test_source_snapshots']
probe=root/'obj/test_module_generation_probe'
manifest={'source_commit':pin,'root':str(root),'command':command,'PATH':env['PATH'],'initial_status':'clean','initial_free_bytes':shutil.disk_usage(root).free,'probe_sha256':sha(probe),'cc_version':subprocess.check_output(['/usr/bin/cc','--version'],text=True)}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('START full source snapshots',flush=True)
start=time.monotonic()
with (out/'gate.log').open('wb') as log:result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
after=snapshot()
(out/'source-after.json').write_text(json.dumps(after,indent=2)+'\n')
manifest.update(final_free_bytes=shutil.disk_usage(root).free,exit_code=result.returncode,seconds=time.monotonic()-start,log_sha256=sha(out/'gate.log'),final_status=git('status','--porcelain'),head_unchanged=git('rev-parse','HEAD')==pin,source_unchanged=before==after,probe_unchanged=sha(probe)==manifest['probe_sha256'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('END',result.returncode,flush=True)
raise SystemExit(result.returncode)
