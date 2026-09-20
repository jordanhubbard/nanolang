import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);report=Path(sys.argv[2]);phase='cyclic-neighbor'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
errors=[];counts={}
for domain in ('source','tools','artifacts'):
 name=f'{phase}-{domain}'+('.json' if domain=='artifacts' else '-after.json')
 rows=json.loads((report/name).read_text());counts[domain]=len(rows)
 for name,row in rows.items():
  path=root/name if domain=='source' else Path(row['path']) if domain=='tools' else Path(name)
  expected=row if domain=='source' else row['sha256']
  try:
   if sha(path)!=expected:errors.append(str(path))
  except Exception as e:errors.append(str(path)+': '+str(e))
print(json.dumps({'counts':counts,'errors':errors}));sys.exit(bool(errors))
