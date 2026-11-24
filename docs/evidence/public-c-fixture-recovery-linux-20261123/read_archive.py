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
changed={'Makefile.gnu','docs/ROADMAP.md','src/env.c','src/parser.c','src/type_metadata_queries.inc','tests/support/public_c_fixture_providers.c','tests/test_type_metadata_queries.c','tests/test_public_c_nonfinite_format.py','tests/test_public_c_supported_types.py'}
assert {p for p,v in new.items() if old.get(p)!=v}==changed
assert new['tests/nanoisa/fixtures/enum_values.nano']=='15d653ce16db49ce3fcdf248719b27c970e8925a769fe7fcb8a5dec7bc496f12'
for p,v in old.items():
 if p not in changed:assert new[p]==v,p
folder='journal/public-c-fixture-recovery-20261123/'
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
for group,count in [('core-controls',34),('eval-controls',8),('make-controls',3),('record-iteration-controls',4)]:
 rows=js(group);assert len(rows)==count and all(r['exit']==0 for r in rows),group
for mode in ['normal','san']:
 assert raw(mode+'-original-eval-run.stdout').count(b'  Testing ')==134
 assert raw(mode+'-checker-run.stdout').count(b'  Testing ')==88
assert len(js('repl-controls'))==15 and all(r['exit']==0 for r in js('repl-controls'))
for mode in ['normal','san','production']:
 for kind,n in [('default',34),('private',41)]:
  assert f'Ran {n} tests'.encode() in raw(mode+'-'+kind+str(n)+'.stderr') and b'\nOK\n' in raw(mode+'-'+kind+str(n)+'.stderr')
profile=js('public-c-controls');assert (profile['tests'],profile['failures'],profile['errors'])==(9,0,0)
unit=js('query-unit');assert len(unit['commands'])==6 and all(c['exit']==0 for c in unit['commands'])
api=[c for c in profile['commands'] if any(p.endswith('_api.c') for p in c['args'])];assert len(api)==3
for c in api:
 assert '-Werror' in c['args'] and '-fsanitize=address,undefined' in c['args'] and '-O1' in c['args'] and any(p.endswith('/tests/support/public_c_fixture_providers.c') for p in c['args'])
assert len(js('policy'))==2 and all(r['exit']==0 for r in js('policy'))
for row in js('make-controls'):
 assert row['target'] in row['args'] and row['target'] in ['test-enum-int64-metadata','test-cseed-enum-lists','test-type-metadata-queries']
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
# I verify failures as failures; no timeout record becomes a completed source gate.
def failed_source(row, started, completed, cpu=None):
 assert (row['exit'],row['starts'],row['completions'])==(1,started,completed)
 assert row['timeout']==600 and row['shadow_timeout']=='default10' and not row['timed_out']
 assert row['before']==row['after'] and row['product'] is None
 stream=provider(row['work']+'/stderr');starts=events(stream,'shadow-start');ends=events(stream,'shadow-complete')
 assert len(starts)==started and len(ends)==completed and started==completed+1
 assert all(e['failures']==0 for e in ends)
 assert [(e['name'],e['source'],e['line']) for e in starts[:completed]]==[(e['name'],e['source'],e['line']) for e in ends]
 phase=row['phases'][-1];assert phase['phase']=='supervisor-status' and phase['parent_timeout'] and not phase['completed'] and phase['signal']==14 and phase['deadline_seconds']==10
 image=row.get('image',row['args'][0]);assert h(provider(image))==row['image_sha256']
 if cpu is not None:
  assert row['worker_cpu']==cpu and row['args'][:3]==['/usr/bin/taskset','-c',str(cpu)]
  assert row['shadow_cpu_samples'] and all(s['processor']==cpu for s in row['shadow_cpu_samples'])
 return starts
failed_source(js('inline-query-failure/private-qualified'),1077,1076)
for kind,n in [('private',1077),('ordinary',1295)]:
 row=js(kind+'-qualified');failed_source(row,n,n-1);assert row['shadow_cpu_samples']
 assert all(new[p]==v for p,v in row['after'].items())
terminal=js('worker-admission-terminal');assert not terminal['source_workload_admitted'] and not terminal['candidate_started']
admission=js('cpu-admission');cpu=admission['selected_cpu'];assert admission['admitted'] and len(admission['rows'])==20 and len(admission['sustained'])==2
for kind,n in [('private',1049),('ordinary',1079)]:failed_source(js('baseline-cpu'+str(cpu)+'-'+kind+'-qualified'),n,n-1,cpu)
assert not any('candidate-cpu' in key and key.endswith('-qualified.json') for key in data)
# My baseline API compile failures use exact predecessor Git blobs and real tools.
baseline=js('baseline');assert baseline['source_pin']=='c8ccb9c751c58a276d8d69bc3706fd095d456f22' and len(baseline['sources'])==15
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for path,digest in baseline['sources'].items():
 proc.stdin.write((baseline['source_pin']+':'+path+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert header[1]==b'blob';b=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';assert h(b)==digest and provider(baseline['work']+'/'+path)==b
proc.stdin.close();assert proc.wait()==0
assert len(baseline['commands'])==3
for c in baseline['commands']:
 assert c['exit']==1 and '-Werror' in c['args'] and '-fsanitize=address,undefined' in c['args']
 source=next(p for p in c['args'] if p.endswith('_api.c'));err=provider(baseline['work']+'/'+Path(source).stem+'.stderr');assert b'undefined reference' in err and b'row_field_info' in err and b'ast_has_service_declaration' in err
cycles=js('cyclic-controls');assert len(cycles['commands'])==2 and all(c['exit']==1 and c['prior_output_sha256']==h(b'previous') for c in cycles['commands'])
for i in range(2):
 assert provider(cycles['work']+'/'+str(i)+'.c')==b'previous'
 assert b'[E0038]' in provider(cycles['work']+'/'+str(i)+'.stderr')
# I share the exact function definitions, not behavioral test stubs.
def prior_bytes(path):
 r=next(r for r in prior['files'] if r['path']=='source/'+path);z=(priorpath.parent/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'];b=gzip.decompress(z);assert h(b)==r['sha256'];return b.decode()
original=js('original-pure-queries');inc=data['source/src/type_metadata_queries.inc'].decode()
for path,key,macro in [('src/env.c','row','NL_DEFINE_ROW_FIELD_INFO'),('src/parser.c','service','NL_DEFINE_AST_SERVICE_QUERY')]:
 old=prior_bytes(path);definition=original[key];assert old.count(definition)==1 and definition in inc
 replacement='#define '+macro+'\n#include "type_metadata_queries.inc"\n#undef '+macro+'\n'
 assert data['source/'+path].decode()==old.replace(definition,replacement)
# Original test inventories and assertion bodies remain; only helper binding and
# the independently observed earlier cyclic diagnostic change.
import ast
old_base=ast.parse(prior_bytes('tests/test_public_c_nonfinite_format.py'));new_base=ast.parse(data['source/tests/test_public_c_nonfinite_format.py'])
def method(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
a=method(old_base,'run_cmd');b=method(new_base,'run_cmd');assert len(b.body)==len(a.body)+1 and isinstance(b.body[0],ast.If)
assert [ast.dump(n) for n in a.body]==[ast.dump(n) for n in b.body[1:]]
a.body=b.body=[];assert ast.dump(old_base)==ast.dump(new_base)
a=ast.parse(prior_bytes('tests/test_public_c_supported_types.py'));b=ast.parse(data['source/tests/test_public_c_supported_types.py']);x=method(a,'test_source_opaque_and_record_order_refusals');y=method(b,'test_source_opaque_and_record_order_refusals')
assert ast.dump(x.body[0])==ast.dump(y.body[0]);old_loop=x.body[1].body;new_loop=y.body[1].body;assert len(old_loop)==len(new_loop)
assert [ast.dump(n) for n in old_loop[:-2]]==[ast.dump(n) for n in new_loop[:-2]] and ast.dump(old_loop[-1])==ast.dump(new_loop[-1])
assert isinstance(new_loop[-2],ast.If) and ast.dump(new_loop[-2].orelse[0])==ast.dump(old_loop[-2]);assert 'E0038' in ast.unparse(new_loop[-2])
x.body=y.body=[];assert ast.dump(a)==ast.dump(b)
# I recompute allocated ELF sections for both actual compiler images.
import struct
def allocated(blob):
 assert blob[:6]==b'\x7fELF\x02\x01';offset=struct.unpack_from('<Q',blob,40)[0];size,count,names=struct.unpack_from('<HHH',blob,58);assert size==64
 sections=[struct.unpack_from('<IIQQQQIIQQ',blob,offset+i*size) for i in range(count)];s=sections[names];strings=blob[s[4]:s[4]+s[5]];out={}
 for row in sections:
  if row[2]&2:
   name=strings[row[0]:strings.find(b'\0',row[0])].decode();b=b'\0'*row[5] if row[1]==8 else blob[row[4]:row[4]+row[5]];out[name]=dict(address=row[3],bytes=row[5],sha256=h(b))
 return out
for row in js('loaded-section-comparison'):
 a=allocated(provider(row['old']));b=allocated(provider(row['new']));assert a==row['old_sections'] and b==row['new_sections'];assert [k for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]==['.note.gnu.build-id']
assert js('native-sanitizers')['exit']==0 and b'Ran 4 tests' in raw('native-sanitizers.stderr') and b'\nOK\n' in raw('native-sanitizers.stderr')
assert len(js('vm-routes')['commands'])==4 and all(c['exit']==0 for c in js('vm-routes')['commands'])
guards=js('frozen-guards');assert guards['images']==json.loads(data['baseline/image-guards.json']) and guards['closure']==json.loads(data['baseline/product-closure.json'])
assert not m['acceptance']['ordinary_full_source'] and not m['acceptance']['private_full_source'] and not m['acceptance']['source_worker_admitted'] and not m['acceptance']['candidate_pinned_attempted'] and not m['acceptance']['whole_release']
print(json.dumps(dict(result='PASS',source_pin=m['source_pin'],records=len(data),git_inputs=len(new),native_images=len(images),focused_c_profile=True,full_source_acceptance=False,whole_release=False),indent=2))
