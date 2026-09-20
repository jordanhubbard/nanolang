import hashlib,json,sys
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
results=[]
for mapping in sys.argv[1:]:
 spec=json.loads(Path(mapping).read_text())
 for entry in spec:
  root=Path(entry['root']);report=Path(entry['report']);source=json.loads((report/entry['source']).read_text());tools=json.loads((report/entry['tools']).read_text())
  bad=[p for p,s in source.items() if not (root/p).is_file() or sha(root/p)!=s]
  badtools=[p for p,v in tools.items() if not Path(v['path']).is_file() or sha(v['path'])!=v['sha256']]
  results.append({'root':str(root),'report':str(report),'source_inventory':entry['source'],'tool_inventory':entry['tools'],'sources':len(source),'tools':len(tools),'source_mismatches':bad,'tool_mismatches':badtools})
print(json.dumps(results,indent=2));assert all(not r['source_mismatches'] and not r['tool_mismatches'] for r in results)
