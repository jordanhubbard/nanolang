from pathlib import Path
import hashlib,json,re,sys
for run in sys.argv[1:]:
 p=Path(run);index={}
 for log in p.glob('*.log'):
  for name in re.findall(rb'I retain raw byte conversion artifacts at ([^\n]+)',log.read_bytes()):
   q=Path(name.decode())
   for f in q.rglob('*'):
    if not f.is_file():continue
    b=f.read_bytes();h=hashlib.sha256(b).hexdigest();target=p/'objects'/h
    if not target.exists():target.write_bytes(b)
    index[str(f)]={'sha256':h,'bytes':len(b),'archive':str(target)}
 (p/'retained-inner.json').write_text(json.dumps(index,indent=2)+'\n')
 print(p.name,len(index))
