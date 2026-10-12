from pathlib import Path
import subprocess,json,hashlib
from tests.test_resource_effect_frontend import HEADER
root=Path.cwd();work=Path('/private/tmp/nl51-effect-collections')
cases={
'scalar':('effect E{pass:array<int>->array<int>}','let a:array<int> =handle {perform E.pass([7])} with {pass a->{a}} return (at a 0)'),
'lexical':('effect E{pass:array<int>->int}','return handle {perform E.pass([7])} with {pass a->{return (at a 0)}}'),
'payload':('struct Batch{token:Handle,items:array<int>} effect E{pass:Batch->Batch}','let b:Batch=handle {perform E.pass(Batch{token:Handle{value:7},items:[7]})} with {pass b->{b}} let Batch{token,items}=b assert (== (consume token) 7) return (at items 0)'),
'wrong_element':('effect E{pass:array<int>->array<int>}','let a:array<int> =handle {perform E.pass([true])} with {pass a->{a}} return (at a 0)')}
rows=[]
for name,(decl,body) in cases.items():
 source=work/(name+'.nano');source.write_text(HEADER+decl+'\nfn main()->int{'+body+'}\nshadow main{assert true}\n')
 for route,driver in [('c',root/'bin/nano_virt'),('nano',Path('/private/tmp/nl51-effect-globals/product/final-compiler'))]:
  out=work/(name+'-'+route+'.nvm');out.write_bytes(b'prior artifact');cmd=[str(driver),str(source),'--emit-nvm','-o',str(out)]
  run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
  log=name+'-'+route+'.log';(work/log).write_text(run.stdout)
  rows.append(dict(case=name,route=route,argv=cmd,returncode=run.returncode,prior_preserved=out.read_bytes()==b'prior artifact',log=log))
(work/'baseline.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row['case'],row['route'],row['returncode'],row['prior_preserved'],(work/row['log']).read_text()[-650:])
