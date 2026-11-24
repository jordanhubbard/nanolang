#!/usr/bin/env python3
"""I independently verify my enum interpreter repair and native failure boundary."""
from pathlib import Path
import gzip,hashlib,json,subprocess,re
base=Path(__file__).resolve().parent;m=json.loads((base/'manifest.json').read_text());h=lambda b:hashlib.sha256(b).hexdigest();rows={};images=set()
for r in m['files']:
 z=(base/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'],r['path'];b=gzip.decompress(z);assert len(b)==r['bytes'] and h(b)==r['sha256'];assert r['path'] not in rows;rows[r['path']]=b
 if b.startswith(b'\x7fELF'):
  assert int.from_bytes(b[18:20],'little')==183
  if int.from_bytes(b[16:18],'little') in (2,3):images.add(h(b))
assert {p.resolve() for p in (base/'objects').glob('*.gz')}=={(base/r['artifact']).resolve() for r in m['files'] if r['artifact'].startswith('objects/')}
priorpath=base/m['base_manifest']['path'];assert h(priorpath.read_bytes())==m['base_manifest']['sha256'];prior=json.loads(priorpath.read_text());old={r['path']:r['sha256'] for r in prior['git_sources']};new={}
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for r in m['git_sources']:
 proc.stdin.write((m['source_pin']+':'+r['path']+'\n').encode());proc.stdin.flush();hdr=proc.stdout.readline().split();assert hdr[1]==b'blob';b=proc.stdout.read(int(hdr[2]));assert proc.stdout.read(1)==b'\n';assert b==rows['source/'+r['path']] and h(b)==r['sha256'];new[r['path']]=r['sha256']
proc.stdin.close();assert proc.wait()==0
changed={'Makefile.gnu','docs/EVAL_CARRIER_OWNERSHIP.md','docs/ROADMAP.md','src/eval.c','src/eval/carrier_runtime.c','src/eval/carrier_runtime.h','src/eval/carrier_values.c','src/eval/carrier_values.h','tests/test_carrier_scalar_list_values.c','tests/test_eval_enum_list_values.c','tests/nanoisa/fixtures/carrier_enum_pressure.nano','tests/nanoisa/fixtures/enum_values.nano'}
assert {p for p,v in new.items() if old.get(p)!=v}==changed
assert new['tests/nanoisa/fixtures/enum_values.nano']=='15d653ce16db49ce3fcdf248719b27c970e8925a769fe7fcb8a5dec7bc496f12'
for p,v in old.items():
 if p.startswith(('src_nano/','modules/','stdlib/','tests/')):assert new[p]==v,p
folder='journal/carrier-enum-repair-20261123/'
def raw(name):return rows[folder+name]
def js(name):return json.loads(raw(name+'.json'))
def provider(path):
 marker='/.test_output/';key='journal/'+path.split(marker,1)[1] if marker in path else ''
 return rows[key] if key in rows else rows['provider/'+path]
q=js('repair');assert set(q['modes'])=={'normal','san','production'}
for name,n in [('repair',24),('field-controls',36),('walk-controls',6),('neighbors',79),('make-controls',3),('policy',2)]:
 r=js(name);commands=r['commands'] if name=='repair' else r;assert len(commands)==n and all(c['exit']==0 for c in commands),name
for mode in ['normal','san','production']:
 for subject in ['env','eval','repl','carrier_heap','carrier_runtime','carrier_values']:
  a=next(c['args'] for c in q['commands'] if c['label']==mode+'-compile-'+subject);assert '-Werror' in a and ('-O3' if mode=='production' else '-O1') in a
  if mode=='san':assert '-fsanitize=address,undefined' in a
 for suffix in ['test_carrier_scalar_list_values','test_eval_enum_list_values']:
  err=raw(mode+'-'+suffix+'-run.stderr');assert b'AddressSanitizer:' not in err and b'runtime error:' not in err
 for suffix,n in [('default34',34),('private41',41)]:
  err=raw(mode+'-'+suffix+'.stderr');assert f'Ran {n} tests'.encode() in err and b'\nOK\n' in err
 a=next(c['args'] for c in js('make-controls') if c['label']==mode);assert 'test-carrier-scalar-list-values' in a and 'test-eval-enum-list-values' in a
for mode in ['normal','san']:
 assert raw(mode+'-eval-run.stdout').count(b'  Testing ')==134
 assert raw(mode+'-checker-run.stdout').count(b'  Testing ')==88
 match=re.search(rb'after64=(\d+) after128=(\d+) peak=(\d+)',raw(mode+'-active-run.stdout'));assert match
 a,b,peak=map(int,match.groups());assert a==b and peak<=32
 assert b'delta=0 calls=128.' in raw(mode+'-native-bytes-run.stdout')
# My earlier broad neighbor pass differs from final eval source by formatting only.
assert h(raw('pre-format-pass/compiler'))==js('pre-format-pass/private-qualified')['image_sha256']
neighbor=js('source');early=raw('pre-format-pass/eval.c');final=rows['source/src/eval.c'];assert neighbor['src/eval.c']==h(early)
assert b'\n'.join(s.rstrip() for s in early.splitlines())==b'\n'.join(s.rstrip() for s in final.splitlines())
for p,v in neighbor.items():
 if p!='src/eval.c':assert new[p]==v,p
reference=next(r for r in prior['files'] if r['path']=='journal/carrier-live-pressure-20261123/ordinary-native/stderr')
z=(priorpath.parent/reference['artifact']).read_bytes();assert h(z)==reference['compressed_sha256'];reference_raw=gzip.decompress(z);assert h(reference_raw)==reference['sha256']
def selection(data):
 result=[]
 for line in data.splitlines():
  try:x=json.loads(line)
  except ValueError:continue
  if x.get('event')=='shadow-start':result.append((x['name'],x['source'].split('/src_nano/',1)[-1],x['line']))
 return result
expected=selection(reference_raw);assert len(expected)==1306
for kind in ['private','ordinary']:
 assert selection(raw(kind+'-native/stderr'))==expected
 q=js(kind+'-qualified');assert (q['exit'],q['starts'],q['completions'])==(0,1306,1306);assert not q['timed_out'] and q['timeout']==600 and q['shadow_timeout']=='default10';assert q['before']==q['after']
 for p,v in q['after'].items():assert new[p]==v,p
 for p,v in q['providers'].items():assert h(provider(p))==v,p
 assert h(provider(q['args'][0]))==q['image_sha256']
 e=[]
 for line in raw(kind+'-native/stderr').splitlines():
  try:e.append(json.loads(line))
  except ValueError:pass
 status=next(x for x in e if x.get('phase')=='supervisor-status');assert status['deadline_seconds']==10 and status['completed'] and not status['parent_timeout'] and status['exit_code']==0
 starts=[x for x in e if x.get('event')=='shadow-start'];ends=[x for x in e if x.get('event')=='shadow-complete'];assert len(starts)==len(ends)==1306 and all(x['failures']==0 for x in ends)
 assert [(x['name'],x['source'],x['line']) for x in starts]==[(x['name'],x['source'],x['line']) for x in ends]
 helpname='seed-help' if kind=='private' else 'ordinary-seed-help';assert js(helpname)['exit']==0
# I preserve both original red runs, then distinguish shadow success from native refusal.
for case,n in [('enum',6),('pressure',7)]:
 for c in js(case+'/observed')['commands']:
  assert c['exit']==1 and not c['outer_timeout'] and c['shadow_timeout']=='default10';assert c['starts']==c['completions']==n;assert c['prior_output_sha256']==c['after_output_sha256']
  e=[]
  for line in raw(case+'/'+c['mode']+'.stderr').splitlines():
   try:e.append(json.loads(line))
   except ValueError:pass
  ends=[x for x in e if x.get('event')=='shadow-complete'];assert len(ends)==n and all(x['failures']==0 for x in ends)
  assert b'C compilation failed' in raw(case+'/'+c['mode']+'.stderr')
red=json.loads(rows['journal/carrier-enum-lists-20261123/observed.json']);assert all(c['exit']==1 and c['starts']==c['completions']==6 and c['prior_output_sha256']==c['after_output_sha256'] for c in red['commands'])
red=json.loads(rows['journal/carrier-enum-pressure-20261123/observed.json']);c=next(c for c in red['commands'] if c['mode']=='private');assert c['exit']==1 and c['starts']==4 and c['completions']==3
assert b'I cannot trace active interpreter roots.' in rows['journal/carrier-enum-pressure-20261123/private.stderr']
guards=js('frozen-guards');assert guards['images']==json.loads(rows['baseline/image-guards.json']);assert guards['closure']==json.loads(rows['baseline/product-closure.json'])
assert not m['acceptance']['ordinary_native_enum_pipeline'] and not m['acceptance']['default_aggregate_ownership'] and not m['acceptance']['whole_release']
print(json.dumps(dict(result='PASS',source_pin=m['source_pin'],records=len(rows),git_inputs=len(new),native_images=len(images),native_enum_acceptance=False,whole_release=False),indent=2))
