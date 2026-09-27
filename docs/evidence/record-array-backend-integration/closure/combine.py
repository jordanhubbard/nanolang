from pathlib import Path
import hashlib,json,os,shutil
root=Path('/home/jkh/nanolang-qualification');out=root/'backend-integration-a1bb-combined-seal';out.mkdir();(out/'reports').mkdir();(out/'objects').mkdir()
seals={'linux':root/'backend-integration-a1bb-linux/seal','puck':root/'backend-integration-a1bb-puck-download/seal'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
reports={};objects={};histories={}
def add_report(p,name):
 dest=out/'reports'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);reports[name]={'sha256':sha(dest),'bytes':dest.stat().st_size}
for host,seal in seals.items():
 idx=json.loads((seal/'report-sha256.json').read_text());store=json.loads((seal/'artifact-store.json').read_text());histories[host]=json.loads((seal/'summary.json').read_text())
 for n,r in idx.items():
  p=seal/'reports'/n;assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'];add_report(p,host+'/'+n)
 for n in ('report-sha256.json','artifact-store.json','summary.json'):add_report(seal/n,host+'/'+n)
 for h,r in store.items():
  p=seal/'objects'/h;assert sha(p)==h and p.stat().st_size==r['bytes'];dest=out/'objects'/h
  if not dest.exists():os.link(p,dest)
  objects[h]={'bytes':r['bytes'],'path':str(dest)}
extras={
 'linux-archive-verification.json':root/'backend-integration-a1bb-linux/archive-verification.json',
 'linux-durable-remote-verification.json':root/'backend-integration-a1bb-linux/durable-remote-verification.json',
 'linux-copy-verification.json':root/'backend-integration-a1bb-linux-copy-verification.json',
 'puck-archive-verification.json':root/'backend-integration-a1bb-puck-download/archive-verification.json',
 'puck-durable-local-verification.json':root/'backend-integration-a1bb-puck-download/local-verification.json',
 'combine.py':Path(__file__)}
for n,p in extras.items():add_report(p,'closure/'+n)
summary={'source':'a1bb1cb85e5551f9e5bb10491f1608877ef6b7b5','canonical_base':'87e0cef0e2473b5a9987615d15f8a38fe2a0e445','reports':len(reports),'objects':len(objects),'object_bytes':sum(x['bytes'] for x in objects.values()),'references':sum(x['references'] for x in histories.values()),'equal_pairs':sum(x['equal_pairs'] for x in histories.values()),'histories':histories,'scope':'Fresh current-main ordinary integration: Linux17 PASS; Puck16 original PASS plus retained Apple package failure and separate configuration+two-method supported-package PASS. Original sanitizer qualification remains separately attributed. Public/source/full graphs remain required.'}
for n,x in [('report-sha256.json',reports),('artifact-store.json',objects),('summary.json',summary)]: (out/n).write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='histories'}))
