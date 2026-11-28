#!/usr/bin/env python3
"""I independently check source identity, native outcomes and retained limits."""
from pathlib import Path
import gzip, hashlib, json, subprocess
base=Path(__file__).resolve().parent
m=json.loads((base/'manifest.json').read_text());h=lambda b:hashlib.sha256(b).hexdigest();data={};images=set()
for row in m['files']:
 z=(base/row['artifact']).read_bytes();assert h(z)==row['compressed_sha256'],row['path'];b=gzip.decompress(z);assert len(b)==row['bytes'] and h(b)==row['sha256'];assert row['path'] not in data;data[row['path']]=b
 if b.startswith(b'\x7fELF'):
  assert int.from_bytes(b[18:20],'little')==183
  if int.from_bytes(b[16:18],'little') in (2,3):images.add(h(b))
assert {p.resolve() for p in (base/'objects').glob('*.gz')}=={(base/r['artifact']).resolve() for r in m['files'] if r['artifact'].startswith('objects/')}
priorpath=base/m['base_manifest']['path'];assert h(priorpath.read_bytes())==m['base_manifest']['sha256'];prior=json.loads(priorpath.read_text());old={r['path']:r['sha256'] for r in prior['git_sources']}
new={};proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for row in m['git_sources']:
 proc.stdin.write((m['source_pin']+':'+row['path']+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert header[1]==b'blob';b=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';assert b==data['source/'+row['path']] and h(b)==row['sha256'];new[row['path']]=row['sha256']
 if row['path'] not in old:
  proc.stdin.write((prior['source_pin']+':'+row['path']+'\n').encode());proc.stdin.flush();previous=proc.stdout.readline().split()
  if previous[1]==b'blob':
   previous_bytes=proc.stdout.read(int(previous[2]));assert proc.stdout.read(1)==b'\n';old[row['path']]=h(previous_bytes)
  else:assert previous[1]==b'missing'
proc.stdin.close();assert proc.wait()==0
changed={'Makefile.gnu','docs/ROADMAP.md','docs/SOURCE_ENUM_VALUES.md','src/c_backend.c','src/env.c','src/eval.c','src/module.c','src/nanolang.h','src/nanovirt/codegen.c','src/parser.c','src/transpiler.c','src/transpiler_iterative_v3_twopass.c','src/typechecker.c','tests/test_cseed_enum_lists.py','tests/test_enum_int64_metadata.c','tests/nanoisa/fixtures/enum_int64_values.nano'}
assert {p for p,v in new.items() if old.get(p)!=v}==changed
assert new['tests/nanoisa/fixtures/enum_values.nano']=='15d653ce16db49ce3fcdf248719b27c970e8925a769fe7fcb8a5dec7bc496f12'
for p,v in old.items():
 if p not in changed:assert new[p]==v,p
folder='journal/native-enum-lists-20261123/'
def raw(p):return data[folder+p]
def js(p):return json.loads(raw(p+'.json'))
def provider(p):
 marker='/.test_output/';key='journal/'+p.split(marker,1)[1] if marker in p else ''
 return data[key] if key in data else data['provider/'+p]
def events(b,event):
 out=[]
 for line in b.splitlines():
  try:e=json.loads(line)
  except ValueError:continue
  if e.get('event')==event:out.append(e)
 return out
assert all(new[p]==v for p,v in js('final-source-identity').items())
q=js('build');assert set(q['modes'])=={'normal','san','production'}
for mode in q['modes']:
 flags=q['modes'][mode]['flags'];assert '-Werror' in flags and ('-O3' if mode=='production' else '-O1') in flags
 if mode=='san':assert '-fsanitize=address,undefined' in flags
 for source in ['typechecker','transpiler','parser','env','eval','module','c_backend','nanovirt_codegen']:
  assert h(provider(q['modes'][mode]['mapping'][source+'.o']))
for group,count in [('core-controls',34),('eval-controls',8),('make-controls',2),('record-iteration-controls',4)]:
 rows=js(group);assert len(rows)==count and all(r['exit']==0 for r in rows),group
for mode in ['normal','san']:
 assert raw(mode+'-original-eval-run.stdout').count(b'  Testing ')==134
 assert raw(mode+'-checker-run.stdout').count(b'  Testing ')==88
assert len(js('repl-controls'))==15 and all(r['exit']==0 for r in js('repl-controls'))
for mode in ['normal','san','production']:
 for kind,n in [('default',34),('private',41)]:
  assert f'Ran {n} tests'.encode() in raw(mode+'-'+kind+str(n)+'.stderr') and b'\nOK\n' in raw(mode+'-'+kind+str(n)+'.stderr')
profile=js('public-c-controls');assert (profile['tests'],profile['failures'],profile['errors'])==(9,4,0)
err=raw('public-c-controls.stderr');assert err.count(b"undefined reference to `row_field_info'")==3 and err.count(b"undefined reference to `ast_has_service_declaration'")==3 and b'E0038' in err and b'cyclic by-value type layouts' in err
assert not m['acceptance']['complete_original_c_profile_regressions']
refusals=js('public-c-failure-boundaries');assert len(refusals)==9 and all(r['exit']==1 and r['output_sha256']==h(b'previous') for r in refusals)
assert len(js('policy'))==2 and all(r['exit']==0 for r in js('policy'))
for row in js('make-controls'):
 assert row['target'] in row['args'] and row['target'] in ['test-enum-int64-metadata','test-cseed-enum-lists']
for row in js('suites'):
 assert row['exit']==0
 assert b'Ran 21 tests' in raw(row['mode']+'-suite.stderr') and b'\nOK\n' in raw(row['mode']+'-suite.stderr')
 assert h(provider(row['image']))==h(provider(q['images'][row['mode']]))
 witnesses=[(k,b) for k,b in data.items() if k.startswith('provider/'+row['work']+'/') and k.endswith('/result.json')];assert len(witnesses)==24,len(witnesses)
 for path,b in witnesses:
  report=json.loads(b);directory=path.rsplit('/',1)[0];assert report['shadow_timeout']=='default10';assert report['compiler_sha256']==h(provider(row['image']));assert report['source_sha256']==h(data[directory+'/main.nano'])
  output=data[directory+('/program.c' if report['source_only'] else '/program')];assert h(output)==report['output_sha256']
  stream=data[directory+'/compiler.stderr'];starts=events(stream,'shadow-start');ends=events(stream,'shadow-complete')
  if report['reject']:
   assert report['exit']!=0 and output==b'I retain prior enum-list output.\n'
   if 'distinct' in directory or 'wrong_parameter' in directory:assert b'type mismatch' in stream.lower() and not starts
  else:
   assert report['exit']==0 and report['native_exit']==(1 if report['native_failure'] else 0)
   if not report['source_only']:
    assert len(starts)==len(ends)>0 and all(e['failures']==0 for e in ends)
    if '/test_existing_enum_values-' in directory:assert len(ends)==6
    if '/test_enum_pressure_and_free-' in directory:assert len(ends)==7
   else:assert not starts
   if report['source_only']:assert report['consumer_exit']==0 and '-Werror' in report['consumer_args'] and '-pedantic-errors' in report['consumer_args']
   assert b'AddressSanitizer:' not in data[directory+'/native.stderr'] and b'runtime error:' not in data[directory+'/native.stderr']
reference=next(r for r in prior['files'] if r['path']=='journal/carrier-enum-repair-20261123/private-native/stderr')
z=(priorpath.parent/reference['artifact']).read_bytes();assert h(z)==reference['compressed_sha256'];reference_bytes=gzip.decompress(z);assert h(reference_bytes)==reference['sha256'];expected=[(e['name'],e['source'],e['line']) for e in events(reference_bytes,'shadow-start')];assert len(expected)==1306
for row in js('suites'):
 path=next(k for k in data if k.startswith('provider/'+row['work']+'/test_source_c_signed64_members-') and k.endswith('/c99.json'));result=json.loads(data[path]);assert result['exit']==result['native_exit']==0 and '-std=c99' in result['args'] and '-pedantic-errors' in result['args']
for kind in ['ordinary','private']:
 row=js(kind+'-qualified');assert (row['exit'],row['starts'],row['completions'])==(0,1306,1306);assert row['timeout']==600 and row['shadow_timeout']=='default10' and not row['timed_out'];assert row['before']==row['after'];assert all(new[p]==v for p,v in row['after'].items());assert h(provider(row['args'][0]))==row['image_sha256'];assert h(provider(row['work']+'/seed'))==row['product']['sha256'];assert js(kind+'-seed-help')['exit']==0
 stderr=provider(row['work']+'/stderr');starts=events(stderr,'shadow-start');ends=events(stderr,'shadow-complete');assert len(starts)==len(ends)==1306 and all(e['failures']==0 for e in ends)
 assert [(e['name'],e['source'],e['line']) for e in starts]==[(e['name'],e['source'],e['line']) for e in ends]==expected
row=js('native-sanitizers');assert row['exit']==0 and row['environment']['ASAN_OPTIONS']=='detect_leaks=1';assert '-fsanitize=address,undefined' in row['environment']['NANO_CFLAGS'];assert b'Ran 4 tests' in raw('native-sanitizers.stderr') and b'\nOK\n' in raw('native-sanitizers.stderr')
assert all(p.startswith(js('native-sanitizers')['work']+'/') for p in js('native-sanitizer-images'))
assert all(v['asan'] and v['ubsan'] for v in js('native-sanitizer-images').values()) and len(js('native-sanitizer-images'))==4
assert len(js('vm-routes')['commands'])==4 and all(c['exit']==0 for c in js('vm-routes')['commands'])
guards=js('frozen-guards');assert guards['images']==json.loads(data['baseline/image-guards.json']);assert guards['closure']==json.loads(data['baseline/product-closure.json'])
assert m['acceptance']['native_enum_lists'] and not m['acceptance']['prior_output_on_native_link_failure'] and not m['acceptance']['default_aggregate_ownership'] and not m['acceptance']['whole_release']
print(json.dumps(dict(result='PASS',source_pin=m['source_pin'],records=len(data),git_inputs=len(new),native_images=len(images),whole_release=False),indent=2))
