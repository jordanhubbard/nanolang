#!/usr/bin/env python3
"""I verify checked record-list loops and tested native names, without release acceptance."""
from pathlib import Path
import ast,gzip,hashlib,json,subprocess
base=Path(__file__).resolve().parent
m=json.loads((base/'manifest.json').read_text());h=lambda b:hashlib.sha256(b).hexdigest()
p=base/m['base_manifest']['path'];assert h(p.read_bytes())==m['base_manifest']['sha256'];prior=json.loads(p.read_text())
rows={};images=set()
for r in m['files']:
 z=(base/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'];b=gzip.decompress(z);assert h(b)==r['sha256'] and len(b)==r['bytes'],r['path'];assert r['path'] not in rows;rows[r['path']]=b
 if b.startswith(b'\x7fELF') and int.from_bytes(b[16:18],'little') in (2,3):
  assert int.from_bytes(b[18:20],'little')==183;images.add(h(b))
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for r in m['git_sources']:
 proc.stdin.write((m['source_pin']+':'+r['path']+'\n').encode());proc.stdin.flush();hdr=proc.stdout.readline().split();assert hdr[1]==b'blob';b=proc.stdout.read(int(hdr[2]));assert proc.stdout.read(1)==b'\n';assert h(b)==r['sha256'] and b==rows['source/'+r['path']],r['path']
proc.stdin.close();assert proc.wait()==0
old={r['path']:r['sha256'] for r in prior['git_sources']};new={r['path']:r['sha256'] for r in m['git_sources']}
allowed={'Makefile.gnu','src/main.c','src/env.c','src/eval.c','src/nanolang.h','src/typechecker.c','src/transpiler_iterative_v3_twopass.c','tests/test_cseed_result_list_payloads.py','tests/test_record_list_iteration.c','tests/fixtures/record_list_iteration.nano','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md'}
assert {p for p,v in new.items() if old.get(p)!=v}==allowed
assert old['src/nanoisa/nvm2c.c']==new['src/nanoisa/nvm2c.c']
def oldbytes(path):return subprocess.check_output(['git','show',prior['source_pin']+':'+path],cwd=base)
header=rows['source/src/nanolang.h'];previous=oldbytes('src/nanolang.h')
assert header.replace(b'const char *get_record_list_element_name(ASTNode *expr, Environment *env);\n',b'')==previous
# Every original regression method retains its assertions, command budgets and negatives.
def methods(b):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(b)) if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
a=methods(oldbytes('tests/test_cseed_result_list_payloads.py'));b=methods(rows['source/tests/test_cseed_result_list_payloads.py']);assert len(a)==4 and len(b)==7 and all(b[k]==v for k,v in a.items())
def js(name):return json.loads(rows['journal/'+name+'.json'])
assert js('checker-cli')['exit']==1 and js('checker-cli')['prior_preserved']
assert b'checked=1' in rows['journal/checker-metadata-run.stdout']
assert js('runtime-prior')['prior_preserved']
assert all(r['exit']==0 for r in js('native-stage'))
assert js('first-qualified')['commands'][0]['exit']==2 and b'dyn_array_free' in rows['journal/normal-fresh.stderr']
assert js('san-diagnostic-qualified')['commands'][0]['exit']==2 and b'heap-use-after-free' in rows['journal/san-first.stderr']
assert b'type_check_root_shadows' not in rows['journal/unchecked-shadow-unit.c'] and b'type_check_root_shadows' in rows['source/tests/test_record_list_iteration.c']
assert all(r['exit']==0 for r in js('typed-qualified')['commands'])
assert all(r['exit']==0 for r in js('neighbors')) and len(js('neighbors'))==8
for mode in ['normal','san']:
 assert rows['journal/'+mode+'-eval-run.stdout'].count(b'  Testing ')==134
 assert rows['journal/'+mode+'-checker-run.stdout'].count(b'  Testing ')==88
for name,count in [('current-qualified',5),('final-qualified',6),('names-qualified',7),('instrumented-qualified',7)]:
 q=js(name);assert(q['tests'],q['failures'],q['errors'],q['success'])==(count,0,0,True)
for name in ['names-qualified','instrumented-qualified']:assert js(name)['test_source_sha256']==new['tests/test_cseed_result_list_payloads.py']
assert js('long-name-before')['exit']==-6 and js('long-name-before')['prior_preserved']
assert b'__strncpy_chk' in rows['journal/long-name-gdb.stdout']
assert js('self-shadow-original')['exit']==0 and js('self-shadow-execute')['exit']==0
assert all(r['exit']==0 for r in js('composite-qualified'))
for prefix in ['current','final','names','instrumented']:
 for r in js(prefix+'-commands'):
  for inp in r.get('inputs',[]):
   if inp:assert h(rows['journal/'+prefix+'-inputs/'+inp['sha256']])==inp['sha256']
  if r.get('product'):assert h(rows['journal/'+prefix+'-inputs/'+r['product']['sha256']])==r['product']['sha256']
for suffix,folder in [('', 'full-source'),('-final','full-source-final')]:
 q=js('full-source'+suffix+'-qualified');assert q['exit']==0 and not q['timed_out'] and q['seconds']<600 and q['timeout']==600 and q['shadow_timeout']=='default';assert q['before']==q['after']
 assert h(rows['journal/'+folder+'/seed'])==q['product']['sha256'];assert h(rows['provider/'+q['args'][0]])==q['compiler_sha256']
 events=[json.loads(s) for s in rows['journal/'+folder+'.stderr'].decode().splitlines() if s.startswith('{"event":"shadow-')];starts=[e for e in events if e['event']=='shadow-start'];ends=[e for e in events if e['event']=='shadow-complete'];assert len(starts)==len(ends)==1306 and all(e['failures']==0 for e in ends);assert[(e['source'],e['name'],e['line']) for e in starts]==[(e['source'],e['name'],e['line']) for e in ends]
 # Earlier stage preceded the native-name repair. Final stage binds the final guarded sources.
 if suffix:
  assert all(new[path]==digest for path,digest in q['after'].items())
assert all(r['exit']==0 and r['byte_identical'] and h(rows['journal/'+Path(r['original']).name])==h(rows['journal/'+Path(r['confirmed']).name]) for r in js('confirmation')['commands'])
assert all(new[p]==v for p,v in js('confirmation')['source'].items())
assert js('seed-final-help')['exit']==0 and b'Compiler' in rows['journal/seed-final-help.stdout']
assert js('policy')['exit']==0 and b'3468 static fixture fragments; 0 reserved names' in rows['journal/policy.stdout']
assert js('final-image-guards')==json.loads(rows['baseline/image-guards.json'])
assert js('reconstructions')['derived_not_original_snapshots']
assert h(rows['journal/reconstructed-first-checker.c'])==js('reconstructions')['checker_sha256']
assert h(rows['journal/reconstructed-pre-names-emitter.c'])==js('reconstructions')['emitter_sha256']
assert m['acceptance']==dict(record_list_iteration=True,complete_tested_native_names=True,first_compiler_source_stage=True,default_activation=False,raw_stage=False,paired_bootstrap=False,whole_release=False)
print(f'PASS {len(rows)} records / {len(new)} exact Git inputs / {len(images)} distinct native ELF images; final 1306 shadows; default/raw/platform acceptance open')
