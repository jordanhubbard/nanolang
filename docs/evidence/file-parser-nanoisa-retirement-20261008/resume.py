import os,json,time,subprocess,hashlib
from pathlib import Path
root=Path('/Users/jordanh/Src/nanolang');out=Path('/private/tmp/nanolang-parser-retirement-evidence-20261008')
env=dict(os.environ);env['PATH']='/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:'+env['PATH'];env['NANOLANG_ROOT']=str(root)
source=out/'parser.nano';assert hashlib.sha256(source.read_bytes()).hexdigest()==json.loads((out/'inputs.json').read_text())['source_sha']
rows=[]
for name in ['nanoc_stage1','nanoc_stage2']:
 exe=out/(name+'-resumed')
 for phase,args in [('compile',[root/'bin'/name,source,'-o',exe]),('execute',[exe])]:
  start=time.monotonic();r=subprocess.run(list(map(str,args)),cwd=root,env=env,capture_output=True,timeout=600)
  (out/(name+'-'+phase+'-resumed.log')).write_bytes(r.stdout+b'\n'+r.stderr)
  rows.append(dict(name=name,phase=phase,args=list(map(str,args)),exit_code=r.returncode,seconds=time.monotonic()-start))
  (out/'resumed-manifest.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,phase,r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
  if phase=='execute':assert r.stdout==b'publisher:1:5:retained\n',r.stdout
