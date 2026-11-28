#!/usr/bin/env python3
"""I verify bounded current-source observations without re-running them."""
from pathlib import Path
import hashlib,gzip,json
here=Path(__file__).resolve().parent
sha=lambda data:hashlib.sha256(data).hexdigest()
m=json.loads((here/'manifest.json').read_text());base=here/m['base_manifest'];assert sha(base.read_bytes())==m['base_manifest_sha256'];b=json.loads(base.read_text());assert b['source_commit']==m['source_commit']
files={}
for r in m['files']:
 c=(here/r['artifact']).read_bytes();assert sha(c)==r['compressed_sha256'];raw=gzip.decompress(c);assert len(raw)==r['bytes'] and sha(raw)==r['sha256'];assert r['path'] not in files;files[r['path']]=raw
for r in b['files']:
 if r['path']=='observation/final-guards.json':base_guards=json.loads(gzip.decompress((base.parent/r['artifact']).read_bytes()))
guards=json.loads(files['guards.json']);assert guards==base_guards
rows=json.loads(files['result.json']);assert len(rows)==20 and len({r['name'] for r in rows})==20
assert sum(r['timed_out'] for r in rows)==1
assert not any(r['returncode']==0 for r in rows)
for r in rows:
 assert r['timeout']==120 and r['shadow_timeout']==10 and r['prior_preserved']
 assert sha(files[r['name']+'/input.nano'])==r['source_sha256']
 assert files[r['name']+'/program.nvm']==b'prior-output\n'
 if r['timed_out']:assert r['name']=='test_resource_effect_checker' and r['returncode']==-9 and r['seconds']>=120
assert b'Exported functions count: 0' in files['test_module_introspection/stderr']+files['test_module_introspection/stdout']
print(json.dumps(dict(status='PASS',records=len(files),cases=20,refusals=19,timeouts=1,guards='PASS'),indent=2))
