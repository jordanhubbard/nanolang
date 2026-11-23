#!/usr/bin/env python3
"""I verify bounded private metadata without accepting default or platform ownership."""
from pathlib import Path
import gzip,hashlib,json,re,subprocess
base=Path(__file__).resolve().parent;m=json.loads((base/'manifest.json').read_text());h=lambda b:hashlib.sha256(b).hexdigest();p=base/m['base_manifest']['path'];assert h(p.read_bytes())==m['base_manifest']['sha256'];prior=json.loads(p.read_text());rows={};images=set()
for r in m['files']:
 z=(base/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'];b=gzip.decompress(z);assert len(b)==r['bytes'] and h(b)==r['sha256'],r['path'];assert r['path'] not in rows;rows[r['path']]=b
 if b.startswith(b'\x7fELF') and int.from_bytes(b[16:18],'little') in (2,3):
  assert int.from_bytes(b[18:20],'little')==183;images.add(h(b))
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def gitbytes(pin,path):
 proc.stdin.write((pin+':'+path+'\n').encode());proc.stdin.flush();hdr=proc.stdout.readline().split();assert hdr[1]==b'blob';b=proc.stdout.read(int(hdr[2]));assert proc.stdout.read(1)==b'\n';return b
for r in m['git_sources']:
 b=gitbytes(m['source_pin'],r['path']);assert h(b)==r['sha256'] and b==rows['source/'+r['path']]
old={r['path']:r['sha256'] for r in prior['git_sources']};new={r['path']:r['sha256'] for r in m['git_sources']};allowed={'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md','src/eval/carrier_heap.c','src/eval/carrier_runtime.c','tests/test_eval_carrier_collection_scratch.c','tests/test_eval_carrier_native_bytes.c','tests/test_eval_carrier_heap_compaction.c'};assert {p for p,v in new.items() if old.get(p)!=v}==allowed
for path in ['src/eval.c','src/env.c','src/typechecker.c','src/nanolang.h','src/nanoisa/nvm2c.c','src/eval/carrier_values.c','tests/test_eval.c','tests/test_eval_carrier_heap_faults.c','tests/test_eval_carrier_heap.c','tests/fixtures/carrier_active_roots.nano']:assert old[path]==new[path]
def js(name,prefix='journal/'):return json.loads(rows[prefix+name+'.json'])
def pathbytes(path):
 marker='/.test_output/carrier-collection-scratch-20261123/'
 return rows['journal/'+path.split(marker,1)[1]] if marker in path else rows['provider/'+path]
def native(text):
 q=re.search(rb'positive=(\d+) before=(\d+) after64=(\d+) after128=(\d+) delta=(-?\d+) calls=(\d+)',text);assert q;positive,before,middle,last,delta,calls=map(int,q.groups());assert positive>=2097152 and calls==128 and delta==last-middle;return delta
# My new permanent native gate preserves the complete original workload body.
q=js('isolated');driver=next(x for r in q['commands'] for x in r['args'] if x.endswith('native-allocation-driver.c'));original=rows['provider/'+driver];permanent=rows['source/tests/test_eval_carrier_native_bytes.c'];needle=b'static size_t first_bytes';normalized=original
root=Path(driver).parents[2]
for oldpath,newpath in [('src/eval/carrier_runtime.h','../src/eval/carrier_runtime.h'),('src/tracing.h','../src/tracing.h'),('tests/test_eval.c','test_eval.c')]:normalized=normalized.replace(str(root/oldpath).encode(),newpath.encode())
assert permanent[permanent.index(needle):].removesuffix(b'\n#endif\n')==normalized[normalized.index(needle):];assert b'last_bytes>middle_bytes+1024' in original
assert [r['exit'] for r in js('first-isolated')['commands']]==[0,1];assert b'undefined reference' in rows['journal/first-isolate-link.stderr']
assert [r['exit'] for r in q['commands']]==[0,0,94] and native(rows['journal/isolate-run.stdout'])==1584
assert [r['exit'] for r in js('isolated-compacted')['commands']]==[0,0,0] and native(rows['journal/isolated-compacted-run.stdout'])==0
original_plan=js('original-native-plan');a=list(next(r['command'] for r in original_plan['commands'] if r['label']=='normal-allocation-link'));b=list(next(r['args'] for r in js('isolated-compacted')['commands'] if r['label']=='isolated-compacted-link'));a[a.index('-o')+1]='OUTPUT';b[b.index('-o')+1]='OUTPUT'
for original_object,replacement_name in [(js('isolated')['original_provider'],'retained-runtime-patched.o'),(js('isolated-compacted')['original_provider'],'retained-heap-compacted.o')]:a[a.index(original_object)]=next(x for x in b if x.endswith('/'+replacement_name))
assert a==b

for name in ['isolated','isolated-compacted']:
 for path,digest in js(name)['unchanged_providers'].items():assert h(pathbytes(path))==digest,path
# I changed only the runtime scratch hunk in the retained older runtime; old headers remain exact.
retained='c4e2aa0500e918fb0304a0734d4cedb30e1c0af6';oldruntime=rows['journal/retained-runtime-original.c'];assert oldruntime==gitbytes(retained,'src/eval/carrier_runtime.c')
for name,b in rows.items():
 if name.startswith('journal/retained-tree/') and name.endswith('.h'):assert b==gitbytes(retained,name.removeprefix('journal/retained-tree/'))
def collector(s):return s[s.index(b'bool env_carrier_collect('):s.index(b'void env_carrier_track(',s.index(b'bool env_carrier_collect('))]
patched=rows['journal/retained-runtime-patched.c'];assert collector(patched)==collector(rows['source/src/eval/carrier_runtime.c']);restored=patched.replace(collector(patched),collector(oldruntime))
for a,b in [(b'roots, *host_roots, *collection_roots;',b'roots, *host_roots;'),(b'root_capacity, host_count, collection_capacity;',b'root_capacity, host_count;'),(b'bool collect, collecting;',b'bool collect;'),(b'\n    free(session->collection_roots);',b'')]:restored=restored.replace(a,b)
assert restored==oldruntime
# Both free-site traces are diagnostic failures, with the original executable unchanged.
assert js('observed','sites/')['before']==js('observed','sites/')['after']=='01164f0cec7feee171ad7e1ef6affc31d7f2728b970687f5262af20f79d2d0b6'
assert b'INFERIOR_EXIT {"exit": 94}' in rows['sites/run.stdout'];matches=js('matched-sites','sites/');assert len(matches['new_528_chunks'])==3 and matches['free_events']==323
assert all('env_carrier_collect' in r['last_free']['backtrace'] for r in matches['new_528_chunks'])
assert b'INFERIOR_EXIT {"exit": 94}' in rows['journal/candidate-sites/run.stdout']
assert native(rows['journal/candidate-sites/run.stdout'])==1648
assert js('observed','journal/candidate-sites/')['before']==js('observed','journal/candidate-sites/')['after']
frees=[json.loads(s[9:]) for s in rows['journal/candidate-sites/run.stdout'].decode().splitlines() if s.startswith('FREE_528 ')];assert len(frees)==5 and all('carrier_resize' in r['backtrace'] for r in frees)
assert [r['exit'] for r in js('old-control')]==[0,1];assert b'table_allocations==initial_allocations' in rows['journal/old-control-run.stderr']
assert rows['journal/retained-heap-original.c']==gitbytes(prior['source_pin'],'src/eval/carrier_heap.c')
for name in ['qualified','compacted']:assert [r['exit'] for r in js(name)['commands']]==[0,0]
for path,digest in js('compacted-source').items():assert h(rows['source/'+path])==digest
for mode in ['normal','san']:
 text=rows['journal/'+mode+'-compacted.stdout'];assert native(text)==0
 assert b'six allocation failures' in text and b'changing-key churn without another table allocation' in text
 assert b'failed growth, failed collection, reentrant refusal' in text and b'65536 linked allocations' in text
for r in js('neighbors-compacted'):assert r['exit']==0
assert len(js('neighbors-compacted'))==8
for mode in ['normal','san']:
 assert rows['journal/'+mode+'-eval-run-compacted.stdout'].count(b'  Testing ')==134
 assert rows['journal/'+mode+'-checker-run-compacted.stdout'].count(b'  Testing ')==88
assert all(not r['detect_leaks'] for r in js('neighbors-compacted') if r['label'].startswith('san-') and r['label'].endswith('-run'))
assert all(r['exit']==0 for r in js('built-compilers')['commands'])
for mode in ['baseline','strings']:
 prefix=mode+'-native';q=js(prefix+'-qualified');assert(q['tests'],q['failures'],q['errors'],q['success'])==(7,0,0,True);assert q['test_source_sha256']==new['tests/test_cseed_result_list_payloads.py']
 for r in js(prefix+'-commands'):
  for inp in r.get('inputs',[]):
   if inp:assert h(rows['journal/'+prefix+'-inputs/'+inp['sha256']])==inp['sha256']
  if r.get('product'):assert h(rows['journal/'+prefix+'-inputs/'+r['product']['sha256']])==r['product']['sha256']
 q=js('full-source-'+mode+'-qualified');assert q['source_pin']==m['source_pin'] and q['exit']==0 and not q['timed_out'] and q['timeout']==600 and q['seconds']<600 and q['shadow_timeout']=='default';assert q['before']==q['after'];assert all(new[p]==v for p,v in q['after'].items());assert h(rows['provider/'+q['args'][0]])==q['compiler_sha256'];assert h(rows['journal/full-source-'+mode+'/seed'])==q['product']['sha256']
 events=[json.loads(s) for s in rows['journal/full-source-'+mode+'.stderr'].decode().splitlines() if s.startswith('{"event":"shadow-')];start=[e for e in events if e['event']=='shadow-start'];end=[e for e in events if e['event']=='shadow-complete'];assert len(start)==len(end)==1306 and all(e['failures']==0 for e in end);assert[(e['source'],e['name'],e['line']) for e in start]==[(e['source'],e['name'],e['line']) for e in end]
 assert js('full-source-'+mode+'-help')['exit']==0 and b'Compiler' in rows['journal/full-source-'+mode+'-help.stdout']
assert js('policy')['exit']==0 and b'3468 static fixture fragments; 0 reserved names' in rows['journal/policy.stdout']
guards=js('final-image-guards');assert guards==json.loads(rows['baseline/image-guards.json']);assert js('first-image-guard-format')=={p:r['sha256'] for p,r in guards.items()}
for path,digest in js('product-closure').items():assert h(rows['provider/'+path])==digest
assert m['acceptance']==dict(private_collection_metadata=True,original_linux_native_byte_criterion=True,original_shadow_fixture=True,first_compiler_source_stage=True,default_activation=False,paired_platform=False,raw_stage=False,whole_release=False)
proc.stdin.close();assert proc.wait()==0
print(f'PASS {len(rows)} records / {len(new)} exact current Git inputs / {len(images)} distinct native ELF images; original heap criterion delta0; two 1306-shadow source stages; default/platform/raw open')
