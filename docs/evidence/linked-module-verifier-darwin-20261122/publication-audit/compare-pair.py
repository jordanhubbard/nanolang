from pathlib import Path
import json,hashlib,sys
linux=Path('.test_output/linked-module-verifier-20261121/linux');mac=Path(sys.argv[1]);out=Path(sys.argv[2]);sha=lambda b:hashlib.sha256(b).hexdigest()
a=json.loads((linux/'manifest.json').read_text());b=json.loads((mac/'manifest.json').read_text());assert a['status']==b['status']=='passed' and a['source_hashes']==b['source_hashes'];record=dict(implementation=b['pin'],source_files=len(a['source_hashes']),sets={})
for group in ['linked-products','unlinked-products']:
 left=linux/group;right=mac/group;x=json.loads((left/'manifest.json').read_text());y=json.loads((right/'manifest.json').read_text());assert x==y,(group,'manifest');rows=[]
 for entry in x['inputs']:
  name=entry['file'];u=(left/'inputs'/name).read_bytes();v=(right/'inputs'/name).read_bytes();assert u==v and sha(u)==entry['sha256'],(group,name)
 # I compare ordinary normalized raw/public reports separately from retained native command diagnostics.
 names=sorted(p.name for p in left.glob('*.txt'));assert names==sorted(p.name for p in right.glob('*.txt'))
 for name in names:
  u=(left/name).read_bytes();v=(right/name).read_bytes();assert u==v,(group,name);rows.append(dict(report=name,bytes=len(u),sha256=sha(u)))
 record['sets'][group]=dict(cases=x['cases'],original_products=x['original_products'],reports=rows,all_input_bytes_identical=True,all_normalized_reports_identical=True)
examples=json.loads(Path('.test_output/linked-module-verifier-darwin-20261122/expected-examples.json').read_text());assert examples==json.loads((mac/'original-example-inputs.json').read_text())['source_hashes'];record['original_example_inputs']=examples
out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(implementation=record['implementation'],source_files=record['source_files'],sets={key:dict(cases=v['cases'],reports=len(v['reports']),original_products=v['original_products']) for key,v in record['sets'].items()},all_cross_host_bytes_and_reports_match=True)))
