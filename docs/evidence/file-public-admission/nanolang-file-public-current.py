from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]);out=Path(sys.argv[2]);rows=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for root in sorted(base.glob('nanolang-file-public-*')):
 if not root.is_dir() or not (root/'source-pin.json').exists():continue
 pin=json.loads((root/'source-pin.json').read_text());src=Path(pin['root']);maps=list(root.glob('*-source-after.json'))
 if not maps:continue
 sm=json.loads(maps[-1].read_text());tm=json.loads(next(root.glob('*-tools-after.json')).read_text())
 row={'report':str(root),'source_root':str(src),'head':pin['head'],'sources':len(sm),'tools':len(tm),'source_mismatches':[],'tool_mismatches':[]}
 for p,d in sm.items():
  if sha(src/p)!=d:row['source_mismatches'].append(p)
 for name,v in tm.items():
  if sha(v['path'])!=v['sha256']:row['tool_mismatches'].append(name)
 rows.append(row)
out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows));assert all(not r['source_mismatches'] and not r['tool_mismatches'] for r in rows)
