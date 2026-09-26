from pathlib import Path
import argparse,hashlib,json,time
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--report',required=True);p.add_argument('--output',required=True);args=p.parse_args()
root=Path(args.root).resolve();report=Path(args.report);rows=json.loads((report/'results.json').read_text());phase=rows[-1]['phase'];record={'at':time.time(),'root':str(root),'report':str(report),'phase':phase,'scope':'Current selected source, tool-file and provider-file bytes only; no inference of an entire SDK, system-library closure or deleted historical products.'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for kind in ['source','tools']:
 mp=report/(phase+'-'+kind+'-after.json');values=json.loads(mp.read_text())
 for name,value in values.items():
  path=root/name if kind=='source' else Path(value['path'])
  expected=value if kind=='source' else value['sha256']
  assert sha(path)==expected,(kind,name)
 record[kind]={'entries':len(values),'map':str(mp),'map_sha256':sha(mp)}
mp=report/(phase+'-artifacts.json');values=json.loads(mp.read_text());selected={}
for name,value in values.items():
 path=Path(name)
 if path.resolve().is_relative_to(root):
  assert sha(path)==value['sha256'],name;selected[name]=value['sha256']
assert selected
record['providers']={'entries':len(selected),'map':str(mp),'map_sha256':sha(mp),'selected':selected}
record['phases']=rows;record['status']='PASS';Path(args.output).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('providers','phases')}))
