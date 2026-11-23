#!/usr/bin/env python3
"""I verify scoped native repairs without accepting default ownership or bootstrap."""
from pathlib import Path
import gzip,hashlib,json,subprocess
base=Path(__file__).resolve().parent;m=json.loads((base/'manifest.json').read_text());h=lambda b:hashlib.sha256(b).hexdigest();p=base/m['base_manifest']['path'];assert h(p.read_bytes())==m['base_manifest']['sha256'];prior=json.loads(p.read_text());rows={};images=set()
for r in m['files']:
 z=(base/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'],r['path'];b=gzip.decompress(z);assert len(b)==r['bytes'] and h(b)==r['sha256'],r['path'];assert r['path'] not in rows;rows[r['path']]=b
 if b.startswith(b'\x7fELF') and int.from_bytes(b[16:18],'little') in (2,3):
  assert int.from_bytes(b[18:20],'little')==183;images.add(h(b))
for r in m['git_sources']:
 b=subprocess.check_output(['git','show',m['source_pin']+':'+r['path']],cwd=base);assert h(b)==r['sha256'] and b==rows['source/'+r['path']],r['path']
old={r['path']:r['sha256'] for r in prior['git_sources']};new={r['path']:r['sha256'] for r in m['git_sources']};allowed={'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md','src/transpiler_iterative_v3_twopass.c','tests/test_cseed_result_list_payloads.py'}
for path in new.keys()-old.keys():
 if path not in allowed:
  old[path]=h(subprocess.check_output(['git','show',prior['source_pin']+':'+path],cwd=base))
assert {p for p,v in new.items() if old.get(p)!=v}==allowed
for path in ['src/env.c','src/eval.c','src/typechecker.c','src/nanolang.h','src/nanoisa/nvm2c.c','tests/test_eval.c','tests/test_cseed_shadow_trace.py']:assert old[path]==new[path]
def js(path):return json.loads(rows[path])
for name in ['built.json','repair-built.json','unit-qualified.json']:
 q=js('journal/'+name);q=q['commands'] if isinstance(q,dict) else q;assert all(r['exit']==0 for r in q)
assert len(js('journal/unit-qualified.json'))==5
for name in ['green-qualified.json','baseline-control-qualified.json','instrumented-qualified.json']:
 q=js('journal/'+name);assert(q['tests'],q['failures'],q['errors'],q['success'])==(4,0,0,True);assert q['test_source_sha256']==h(rows['source/tests/test_cseed_result_list_payloads.py'])
q=js('journal/red-qualified.json');assert(q['tests'],q['failures'],q['errors'])==(4,2,0)
q=js('journal/loop-red-qualified.json');assert(q['tests'],q['failures'],q['errors'])==(5,1,0)
experiment=rows['journal/experimental-loop-tests.py'].decode();a=experiment.index('    def test_generated_list_iteration_uses_value_elements');b=experiment.index('    def test_wrong_variant_',a);assert(experiment[:a]+experiment[b:]).encode()==rows['source/tests/test_cseed_result_list_payloads.py']
assert b'Field access requires a struct value' in rows['journal/loop-red-tests.stdout']
assert rows['journal/first-payload-tests.py']==rows['source/tests/test_cseed_result_list_payloads.py']
for name in ['trace-qualified.json','final-trace-qualified.json']:
 q=js('journal/'+name);assert(q['tests'],q['failures'],q['errors'],q['success'])==(3,0,0,True);assert q['test_source_sha256']==h(rows['source/tests/test_cseed_shadow_trace.py'])
for prefix in ['red','green','baseline-control','loop-red','trace','final-trace','instrumented']:
 for r in js('journal/'+prefix+'-commands.json'):
  for inp in r['inputs']:
   if inp:assert h(rows['journal/'+prefix+'-inputs/'+inp['sha256']])==inp['sha256']
  if r.get('product'):assert h(rows['journal/'+prefix+'-inputs/'+r['product']['sha256']])==r['product']['sha256']
# The unsupported --emit-nvm attempt must not become shadow evidence.
assert all(r['exit']==1 and r['prior_preserved'] and not r['events'] for r in js('journal/cli-qualified.json'))
first=js('journal/native-qualified.json');assert[(r['exit']) for r in first]==[1,0,1,1,1,0,0,0,1,1]
final=js('journal/repair-native-qualified.json');assert[r['exit'] for r in final]==[1,0,1,1,0,0,0,0,0,0]
for r in final:
 if 'candidate' in r['label']:
  assert r['exit']==0 and not r['prior_preserved'];assert all(e['failures']==0 for e in r['events']);assert len(r['events']) in [4,6,8,20]
for r in js('journal/executed.json'):
 assert r['exit']==r['expected'];assert h(rows['journal/products/'+Path(r['args'][0]).name])==r['sha256']
f=js('journal/full-source-qualified.json');assert f['exit']==0 and not f['timed_out'] and f['seconds']<600 and f['timeout']==600 and f['shadow_timeout']=='default';assert f['before']==f['after']
for path,digest in f['after'].items():assert h(rows['source/'+path])==digest
assert h(rows['journal/full-source/seed'])==f['product']['sha256'];assert h(rows['provider/'+f['args'][0]])==f['compiler_sha256']
events=[json.loads(s) for s in rows['journal/full-source.stderr'].decode().splitlines() if s.startswith('{"event":"shadow-')];starts=[e for e in events if e['event']=='shadow-start'];ends=[e for e in events if e['event']=='shadow-complete'];assert len(starts)==len(ends)==1306 and all(e['failures']==0 for e in ends);assert[(e['source'],e['name'],e['line']) for e in starts]==[(e['source'],e['name'],e['line']) for e in ends]
assert js('journal/seed-help.json')['exit']==0 and b'Compiler' in rows['journal/seed-help.stdout']
assert js('journal/identifier-policy.json')['exit']==0
for path,r in js('journal/product-identities.json').items():assert h(rows['provider/'+path])==r['sha256']
assert js('journal/final-image-guards.json')==js('baseline/image-guards.json')
# I preserve every provider except the two explicit replacement objects.
built=js('journal/built.json')['commands'];repaired=js('journal/repair-built.json')
for mode in ['baseline','candidate']:
 original=next(r['args'] for r in built if r['label']==mode+'-link');updated=next(r['args'] for r in repaired if r['label']=='repair-'+mode+'-link');a=list(original);b=list(updated);a[a.index('-o')+1]='OUTPUT';b[b.index('-o')+1]='OUTPUT';a=[x for x in a if not x.endswith('/normal/obj/transpiler.o')];b=[x for x in b if not x.endswith('/repaired-transpiler.o')];assert a==b
assert js('journal/instrumented-profile.json')['ASAN_OPTIONS']=='detect_leaks=0'
assert not js('journal/instrumented-profile.json')['whole_provider_sanitized']
assert js('journal/instrumented-built.json')['exit']==0
assert m['acceptance']==dict(changed_emitter_sanitizer=True,result_unwrap_payloads=True,generated_list_direct_calls=True,explicit_string_candidate_cli=True,first_compiler_source_stage=True,record_list_iteration=False,default_activation=False,raw_stage=False,paired_bootstrap=False,whole_release=False)
print(f"PASS {len(rows)} records / {len(new)} exact Git inputs / {len(images)} distinct native ELF images; 1306 compiler shadows; default/raw/platform acceptance open")
