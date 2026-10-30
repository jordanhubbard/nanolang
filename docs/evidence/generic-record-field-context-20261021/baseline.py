from pathlib import Path
import sys,json,time,hashlib,os
r=Path.cwd();sys.path.insert(0,str(r));from tests.native_process import run_captured
from tests import test_native_generic_record_fields as cases
w=Path(__file__).parent
class Capture:
 def check(self,source,*args,**kwargs):self.source=source
c=Capture();cases.NativeGenericRecordLayout.test_direct_field_match_executes(c)
src=w/'unchanged-direct.nano';src.write_text(c.source)
root=Path('/Users/jkh/Src/nanolang-51-harness');providers={k:root/'bin'/k for k in ['nanoc_c','nanoc_stage1','nanoc_stage2']};providers['current-c-hosted']=Path('/private/tmp/nanolang-map-remove-20261021/canonical')
rows=[]
for name,tool in providers.items():
 output=w/(name+'-native');output.write_bytes(b'prior artifact');args=[str(tool),str(src),'-o',str(output)]
 t=time.monotonic();p=run_captured(args,cwd=root,timeout=180);(w/(name+'.stdout')).write_bytes(p.stdout);(w/(name+'.stderr')).write_bytes(p.stderr)
 row=dict(name=name,argv=args,exit=p.returncode,seconds=time.monotonic()-t,prior_preserved=output.read_bytes()==b'prior artifact',sha256=hashlib.sha256(tool.read_bytes()).hexdigest())
 if p.returncode==0:
  q=run_captured([str(output)],cwd=root,timeout=20);row['native_exit']=q.returncode;(w/(name+'-run.stdout')).write_bytes(q.stdout);(w/(name+'-run.stderr')).write_bytes(q.stderr)
 rows.append(row);(w/'baseline.json').write_text(json.dumps(rows,indent=2));print(row, p.stderr.decode(),p.stdout.decode(),flush=True)
