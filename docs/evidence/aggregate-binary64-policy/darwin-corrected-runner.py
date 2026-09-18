from pathlib import Path
import subprocess,os,json,hashlib,time
root=Path('/private/tmp/nanolang-aggregate-policy-darwin-tka5uckp/source')
out=root.parent/'corrected-adjacent';out.mkdir(exist_ok=True)
env={**os.environ,'PATH':'/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','CC':'/usr/bin/clang'}
subprocess.run(['git','fetch','origin','fix/aggregate-binary64-policy'],cwd=root,env=env,check=True)
subprocess.run(['git','cherry-pick','13460195'],cwd=root,env=env,check=True)
files=subprocess.check_output(['git','ls-files','src','src_nano','tests','modules','Makefile.gnu'],cwd=root,text=True).splitlines()+['bin/'+n for n in ['nanoc_c','nanoc_stage1','nanoc_stage2','nano','nano_virt','nano_vm','nvm2c','nanoisa']]
def hashes():return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files if (root/p).is_file()}
before=hashes();(out/'before.json').write_text(json.dumps(before,indent=2))
cmd=['make','-j8','CC=/usr/bin/clang','test-nanovm'];start=time.monotonic()
with (out/'suite.log').open('w') as log:r=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
after=hashes();(out/'after.json').write_text(json.dumps(after,indent=2));status={'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'command':cmd,'exit':r.returncode,'seconds':time.monotonic()-start,'unchanged':before==after};(out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status));raise SystemExit(r.returncode if r.returncode else int(before!=after))
