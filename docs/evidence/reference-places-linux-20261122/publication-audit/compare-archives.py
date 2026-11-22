from pathlib import Path
import sys,json,gzip,hashlib
left,right=map(Path,sys.argv[1:3]);a=json.loads((left/'manifest.json').read_text());b=json.loads((right/'manifest.json').read_text());sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert a['implementation']==b['implementation'] and a['source_hashes']==b['source_hashes']
x=a['verification'];y=b['verification'];assert x['inputs']==y['inputs'] and x['cases']==y['cases'] and x['independent_expectations']==y['independent_expectations']
def products(root,m):
 paths={e['path']:e for e in m['files']};matches=[p[:-len('case-0.bin')] for p in paths if p.endswith('/products/case-0.bin')];assert len(matches)==1
 def read(name):return gzip.decompress((root/paths[matches[0]+name]['artifact']).read_bytes())
 return read
read_a=products(left,a);read_b=products(right,b)
for e in x['inputs']:
 raw=read_a(e['file']);assert raw==read_b(e['file']) and sha(raw)==e['sha256']
names=['actual-c']+[r['name'] for r in x['routes']];assert [r['name'] for r in x['routes']]==[r['name'] for r in y['routes']];reports=[]
for name in names:
 raw=read_a(name+'.reports');assert raw==read_b(name+'.reports');assert len(raw.splitlines())==x['cases'];reports.append(dict(file=name+'.reports',sha256=sha(raw),reports=x['cases']))
print(json.dumps(dict(implementation=a['implementation'],cases=x['cases'],independent_expectations=x['independent_expectations'],source_files=len(a['source_hashes']),all_input_bytes_and_source_hashes_identical=True,all_actual_C_and_five_route_report_bytes_identical=True,reports=reports),indent=2))
