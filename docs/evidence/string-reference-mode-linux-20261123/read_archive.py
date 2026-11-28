#!/usr/bin/env python3
"""I verify opt-in STRING qualification without accepting ordinary ownership."""
from pathlib import Path
import gzip,hashlib,json,re,subprocess
base=Path(__file__).resolve().parent;m=json.loads((base/'manifest.json').read_text());h=lambda b:hashlib.sha256(b).hexdigest()
p=base/m['base_manifest']['path'];assert h(p.read_bytes())==m['base_manifest']['sha256'];prior=json.loads(p.read_text());rows={}
for r in m['files']:
 z=(base/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'],r['path'];b=gzip.decompress(z)
 assert len(b)==r['bytes'] and h(b)==r['sha256'],r['path'];assert r['path'] not in rows;rows[r['path']]=b
for s in m['git_sources']:
 b=subprocess.check_output(['git','show',m['source_pin']+':'+s['path']],cwd=base);assert h(b)==s['sha256'] and b==rows['source/'+s['path']],s['path']
old={r['path']:r['sha256'] for r in prior['git_sources']};new={r['path']:r['sha256'] for r in m['git_sources']}
allowed={'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md','src/eval.c','src/eval/carrier_runtime.c','src/eval/carrier_runtime.h','tests/test_eval_string_reference_mode.c','tests/fixtures/carrier_string_sibling.nano','tests/fixtures/carrier_string_duplicate.nano','tests/fixtures/carrier_string_async.nano'}
assert {p for p,v in new.items() if old.get(p)!=v}<=allowed
assert old['src/env.c']==new['src/env.c'] and old['src/nanolang.h']==new['src/nanolang.h'] and old['tests/test_eval.c']==new['tests/test_eval.c']
def oldbytes(path):
 r=next(r for r in prior['files'] if r['path']=='source/'+path);return gzip.decompress((p.parent/r['artifact']).read_bytes())
assert rows['source/src/eval.c'].replace(b'env_carrier_scopes_enabled',b'env_carrier_enabled')==oldbytes('src/eval.c')
a=iter(rows['source/Makefile.gnu'].splitlines());assert all(any(v==line for v in a) for line in oldbytes('Makefile.gnu').splitlines())
assert rows['source/tests/fixtures/carrier_string_duplicate.nano']==rows['baseline/duplicate-return.nano']
assert rows['source/tests/fixtures/carrier_string_sibling.nano']==rows['baseline/string-sibling.nano']
def js(p):return json.loads(rows[p])
assert all(r['exit']==0 for r in js('journal/qualified.json')['commands'])
neighbors=js('journal/neighbors.json')['commands'];assert len(neighbors)==12
assert {r['label']:r['exit'] for r in neighbors if r['exit']}=={'duplicate-strings-lsan':1}
assert all(r['detect_leaks'] for r in neighbors if r['label'].endswith('final-unit'))
for profile in ['normal','san']:
 text=rows['journal/'+profile+'-final-unit.stdout'].decode()
 for mode in ['strings','full']:
  assert f'carrier_string_sibling.nano mode={mode} registered=16 released=16 exported=0 live=0.' in text
  assert f'carrier_string_async.nano mode={mode} registered=1339 released=1338 exported=1 live=0.' in text
 assert 'carrier_string_duplicate.nano mode=full registered=6 released=6 exported=0 live=0.' in text
 for mode in ['ordinary','strings']:
  text=rows['journal/'+profile+'-'+mode+'-suite.stdout'].decode();assert text.count('  Testing ')==134 and 'All eval tests passed!' in text
full=js('journal/full-qualified.json')['commands'];assert len(full)==6
assert {r['label']:r['exit'] for r in full if r['exit']}=={'san-ordinary-sibling-run':1}
for profile in ['normal','san']:
 text=rows['journal/'+profile+'-full-run.stdout'].decode();assert text.count('  Testing ')==134 and 'All eval tests passed!' in text
assert b'heap-use-after-free' in rows['journal/san-ordinary-sibling-run.stderr']
assert b'59 byte(s) leaked in 4 allocation(s)' in rows['journal/duplicate-strings-lsan.stderr']
assert b'create_array src/env.c' in rows['journal/duplicate-strings-lsan.stderr']
assert b'Expected variable name' in rows['journal/first-private.stderr']
assert b'cannot find a function named `spawn`' in rows['journal/private.stderr']
assert b'cannot find a variable named `scheduler_run`' in rows['journal/private.stderr']
for path,digest in js('journal/source-identities.json').items():assert h(rows['source/'+path])==digest
for path in m['product_paths']:
 b=rows['provider/'+path];assert b[:4]==b'\x7fELF' and int.from_bytes(b[18:20],'little')==183
assert js('journal/final-image-guards.json')==js('baseline/image-guards.json')
assert m['acceptance']==dict(strings_only_opt_in=True,aggregate_mode_preserved=True,ordinary_sibling_safety=False,default_activation=False,complete_ownership=False,whole_release=False)
print(f"PASS {len(rows)} records / {len(new)} Git inputs / {len(m['product_paths'])} final provider ELF images; STRING-only prerequisite; ordinary ownership remains open")
