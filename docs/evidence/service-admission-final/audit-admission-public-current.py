import json,hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);report=Path(sys.argv[2]);out=Path(sys.argv[3])
source=json.loads((report/'ordinary-source-after.json').read_text());tools=json.loads((report/'ordinary-tools-after.json').read_text())
for name,h in source.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
for name,v in tools.items():assert hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'],name
value={'root':str(root),'selected_endpoint':str(report),'current_sources':len(source),'current_tools':len(tools),'all_hashes_equal':True,'scope':'Named tools and tracked source; public install deliberately rebuilds providers recorded as products'}
out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
