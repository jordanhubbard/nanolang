from pathlib import Path
import hashlib,json,time
base=Path('/home/jkh/nanolang-qualification/vm-effects-20260921/nanolang-record-generated-complete-seal')
start=time.monotonic()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
 return h.hexdigest()
reports=json.loads((base/'report-sha256.json').read_text())
objects=json.loads((base/'artifact-store.json').read_text())
errors=[]
for name,row in reports.items():
 p=base/'reports'/name
 if not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:errors.append('report:'+name)
for key,row in objects.items():
 p=base/'objects'/key
 if not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=key:errors.append('object:'+key)
result={'scope':'Independent retained report and CAS byte integrity only; semantic acceptance and current-main integration remain separate.','reports':len(reports),'objects':len(objects),'object_bytes':sum(x['bytes'] for x in objects.values()),'errors':errors,'seconds':time.monotonic()-start}
Path('/home/jkh/nanolang-qualification/generated-c-root-integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
raise SystemExit(bool(errors))
