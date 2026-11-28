"""I verify the actual collection refusal; I grant no product acceptance."""
from pathlib import Path
import argparse,gzip,hashlib,json,re
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();root=a.root
expected={'result.json': 'c72f41a0d0bdf6cd6e6bfeb9e3d8b5e67a1977612f6d24bfb6e113f20b3132ff', 'job.json': 'e3d8d55ee428ffe3f297d5fbce632cf19903527625fdabcd5fa3685c04c6f0ad', 'artifacts.json': '778c6a46320b615545900708777bc4f1e8cde5f7a22148c2ba9ba5baa5d1038b', 'job.log.gz': '245ade330c593b95a0f18b2aedefa71da91add2c2607f85eeef7f2e82a00f7d6'}
for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
q=json.loads((root/'job.json').read_text());r=json.loads((root/'result.json').read_text());metadata=json.loads((root/'artifacts.json').read_text())
assert (q['id'],q['run_id'],q['head_sha'],q['conclusion'])==(140901037356,46119624230,r['source'],'failure')
for name in ['Build fresh production tools','Check managed STRING publication and exact alias retention','Execute my array repair regression','Execute the complete original cache and worker gates','Verify source and selected tool guards']:
 assert next(s for s in q['steps'] if s['name']==name)['conclusion']=='success',name
assert next(s for s in q['steps'] if s['name']=='Retain terminal logs, products and actual host closures')['conclusion']=='failure'
text=gzip.decompress((root/'job.log.gz').read_bytes()).decode()
text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
text=re.sub(r'^\d{4}-\d\d-\d\dT\S+Z ','',text,flags=re.M)
assert [int(m) for m in re.findall(r'^Ran (\d+) tests? in ',text,re.M)]==[2,4,4,6,42,57,4,132,20,16,13,3]
assert sum([42,57,4,132,20,16,13])==r['original_cache_methods_reported']==284
assert text.count('OK (skipped=4)')==text.count('OK (skipped=22)')==1
assert r['existing_skips']==26 and r['worker_methods_reported']==3 and r['array_methods_reported']==6
assert 'Shape constraints: 3890 passed, 0 failed' in text
assert text.count('Failed to CreateArtifact: Unable to make request: ENOTFOUND')==1
assert 'With the provided path, there will be 164 files uploaded' in text
assert not any(x['name']=='c-repair-original-xcode-27-cache-'+r['source'] for x in metadata['artifacts'])
assert r['artifact_id'] is None and not r['actual_product_tool_archive_available'] and not r['qualified_original_darwin_cache'] and not r['whole_release']
assert r['collector_maximum_attempts']==3 and not r['product_tests_retried_by_collector']
print('I verify reported original284/cache26 skips/3 workers/6 array cases and passed guard steps, then CreateArtifact ENOTFOUND with no artifact. The missing actual product/tool archive prevents verified cache acceptance.')
