from pathlib import Path
import json,hashlib,sys
report=Path(sys.argv[1]);root=Path(json.loads((report/'host.json').read_text())['source_root'])
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
sources=json.loads((report/'source-after.json').read_text())
for p,h in sources.items():assert sha(root/p)==h,p
tools=0;artifacts=0
for p in report.glob('*/tools-after.json'):
 for entry in json.loads(p.read_text()).values():assert sha(entry['path'])==entry['sha256'];tools+=1
for p in report.glob('*/artifacts.json'):
 for rel,h in json.loads(p.read_text()).items():assert sha(p.parent/rel)==h;artifacts+=1
print(json.dumps({'source_root':str(root),'report':str(report),'source_entries':len(sources),'current_tools':tools,'current_artifacts':artifacts,'all_current_match':True},indent=2))
