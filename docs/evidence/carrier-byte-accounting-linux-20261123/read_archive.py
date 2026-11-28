#!/usr/bin/env python3
"""I verify a bounded diagnostic without accepting the retained ordinary gate."""
from pathlib import Path
import gzip,hashlib,json,re,subprocess
base=Path(__file__).resolve().parent
m=json.loads((base/'manifest.json').read_text())
h=lambda b:hashlib.sha256(b).hexdigest()
p=base/m['base_manifest']['path'];assert h(p.read_bytes())==m['base_manifest']['sha256']
prior=json.loads(p.read_text());rows={}
for r in m['files']:
 z=(base/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'],r['path']
 b=gzip.decompress(z);assert h(b)==r['sha256'] and len(b)==r['bytes'],r['path'];assert r['path'] not in rows;rows[r['path']]=b
def js(p):return json.loads(rows[p])
q=js('journal/observed.json');assert q['source']==m['source_pin'];assert [(r['label'],r['exit']) for r in q['commands']]==[('link',0),('run',0)]
first=js('journal/first-observed.json');assert first['commands'][0]['exit']==1
assert b'_GNU_SOURCE' in rows['journal/first-link.stderr']
assert b'error:' not in rows['journal/link.stderr']
original=rows['baseline/native-allocation-driver.c'].decode();s='extern void allocation_sample(unsigned);\nextern void allocation_report(void);\n'+original
s=s.replace('if(chosen && !observed_calls) first_bytes=allocated();','if(chosen && !observed_calls) { first_bytes=allocated(); allocation_sample(0); }')
s=s.replace('if(observed_calls==64)middle_bytes=allocated();if(observed_calls==128)last_bytes=allocated();','if(observed_calls==64){middle_bytes=allocated();allocation_sample(1);}if(observed_calls==128){last_bytes=allocated();allocation_sample(2);}')
s=s.replace('int result=fixture_main();','int result=fixture_main(); allocation_sample(3); allocation_report();')
assert s.encode()==rows['journal/driver.c'];assert 'last_bytes>middle_bytes+1024' in original
text=rows['journal/run.stdout'].decode();samples={int(v[0]):tuple(map(int,v[1:])) for v in re.findall(r'account sample=(\d+) requested=(\d+) usable=(\d+) live=(\d+) missing=(\d+) duplicate=(\d+) overflow=(\d+)',text)}
assert len(samples)==4 and samples[1]==samples[2]
assert samples[3][:3]==(0,0,0)
assert all(v[3:]==(4,0,0) for v in samples.values())
assert 'outstanding epoch=' not in text
assert 'delta=0 calls=128.' in text
assert 'delta=1776 calls=128.' in rows['baseline/final-source-normal-allocation-run.stdout'].decode()
n=js('journal/no-tcache.json');assert n['exit']==0 and n['GLIBC_TUNABLES']=='glibc.malloc.tcache_count=0'
assert 'delta=0 calls=128.' in rows['journal/no-tcache.stdout'].decode()
ident=js('baseline/retained-inputs.json')
for ref in js('journal/provider-inputs.json'):
 b=rows['provider/'+ref['path']];assert h(b)==ref['sha256'] and len(b)==ref['bytes']
 assert any(ref['path'] in group and group[ref['path']]['sha256']==ref['sha256'] for group in ident.values())
assert n['product_sha256']==h(rows['provider/'+n['args'][0]])
image=rows['journal/account-observer'];assert image[:4]==b'\x7fELF' and int.from_bytes(image[18:20],'little')==183
# The retained C shadow driver and NanoLang workload must be the prior exact bytes.
git=lambda path:subprocess.check_output(['git','show',m['source_pin']+':'+path],cwd=base)
for path,digest in q['inputs'].items():
 if path.startswith('.test_output/carrier-byte-accounting-20261123/'):
  assert h(rows['journal/'+Path(path).name])==digest
 elif path.endswith('native-allocation-driver.c'):assert h(rows['baseline/native-allocation-driver.c'])==digest
 else:assert h(git(path))==digest
assert m['acceptance']==dict(diagnostic_only=True,ordinary_original_gate=False,default_ownership=False,whole_release=False)
print(f"PASS {len(rows)} records; instrumented outstanding bytes flat/teardown zero; original +1776 gate remains unaccepted")
