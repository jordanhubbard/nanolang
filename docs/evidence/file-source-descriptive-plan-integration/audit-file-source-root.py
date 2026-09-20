from pathlib import Path
import subprocess,json,hashlib,collections
pin='96495b848';prefix='docs/evidence/file-source-descriptive-plan/'
def blob(n):return subprocess.check_output(['git','show',pin+':'+prefix+n])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads(blob('report-sha256.json'));raw={};data={}
for n,h in manifest.items():
 b=blob(n);assert hashlib.sha256(b).hexdigest()==h,n;raw[n]=b
 if n.endswith('.json'):data[n]=json.loads(b)
index=data['artifact-index.json'];store=index['objects']
for h,row in store.items():assert sha(row['archive'])==h and Path(row['archive']).stat().st_size==row['bytes'],h
for n,h in index['references'].items():assert h in store,n
pairs=[]
for n,d in data.items():
 if n.endswith(('source-before.json','tools-before.json')):
  other=n.replace('-before.json','-after.json')
  if other in data:assert d==data[other],n;pairs.append(n)
selections=[]
for n,d in data.items():
 if n.endswith('-selection.json'):
  assert len(d['expected'])==75 and collections.Counter(d['normalized'])==collections.Counter(d['expected']),n
  directory=n.rsplit('/',1)[0]+'/';compiler=Path(n).name.removesuffix('-selection.json')
  assert raw[directory+compiler+'-run.stdout']==raw[directory+'corpus-c-run.stdout'],n
  assert len(data[directory+'cases.json'])==58
  selections.append(n)
assert len(selections)==12
# I locate the matching immutable download by content, not by a guessed filename.
archives=[]
for p in Path('/tmp').glob('*file-source*.tar.gz'):
 if p.stat().st_size==26507141:
  assert sha(p)=='fd17ffc384d58b9a570bc7caf78720ed846a4dd086edb6900f25ccb8b086ea22';archives.append(str(p))
assert archives
out={'pin':pin,'reports':len(manifest),'artifacts':len(store),'bytes':sum(x['bytes'] for x in store.values()),'references':len(index['references']),'equal_pairs':len(pairs),'exact75_shadow_producer_runs':len(selections),'case_count':58,'puck_archives':archives,'pass':True}
Path('/tmp/file-source-root-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
