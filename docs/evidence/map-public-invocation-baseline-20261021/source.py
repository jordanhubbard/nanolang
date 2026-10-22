from pathlib import Path
import json,os,subprocess,sys,time,hashlib
r=Path.cwd();w=Path(__file__).parent;sys.path.insert(0,str(r));from tests.native_process import run_captured
root=Path('/Users/jkh/Src/nanolang-51-harness');producers=[('seed',root/'bin/nano_virt'),('c-hosted-nano',Path('/private/tmp/nanolang-self-tail-proof-20261021/emitter')),('stage1',root/'bin/nanoc_stage1'),('stage2',root/'bin/nanoc_stage2')];rows=[];pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for _,p in producers}
for name,p in producers:
 m=w/(name+'.nvm')
 for label,args in [('compile',[p,w/'global.nano','--emit-nvm','-o',m]),('verify',[root/'bin/nano_vm','--verify-only',m]),('repeat',[w/'baseline',m])]:
  t=time.monotonic();z=run_captured(list(map(str,args)),cwd=r,env=dict(os.environ,LSAN_OPTIONS=''),timeout=180);(w/(name+'-'+label+'.stdout')).write_bytes(z.stdout);(w/(name+'-'+label+'.stderr')).write_bytes(z.stderr);rows.append(dict(name=name+'-'+label,argv=list(map(str,args)),exit=z.returncode,seconds=time.monotonic()-t));(w/'source-status.json').write_text(json.dumps(rows,indent=2));print(name,label,z.returncode,flush=True)
  if z.returncode:print(z.stderr.decode());break
assert pins=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for _,p in producers}
pins[str(w/'global.nano')]=hashlib.sha256((w/'global.nano').read_bytes()).hexdigest();(w/'source-pins.json').write_text(json.dumps(pins,indent=2))
