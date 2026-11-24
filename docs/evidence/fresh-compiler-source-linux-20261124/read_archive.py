#!/usr/bin/env python3
"""I verify current reference compiler-source gates and the retained setup refusal."""
from pathlib import Path
import gzip,hashlib,json,re,subprocess
BASE=Path(__file__).resolve().parent;REPO=BASE.parents[2]
m=json.loads((BASE/'manifest.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest();rows={r['path']:r for r in m['files']};assert len(rows)==len(m['files']);verified={};images=set()
for row in rows.values():
 p=(BASE/row['artifact']).resolve();assert p.is_relative_to(REPO/'docs/evidence');identity=(row['compressed_sha256'],row['sha256'],row['bytes'])
 if p in verified:assert verified[p]==identity;continue
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):digest.update(b)
 assert digest.hexdigest()==row['compressed_sha256']
 digest=hashlib.sha256();size=0;head=b''
 with gzip.open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):
   if not head:head=b[:64]
   digest.update(b);size+=len(b)
 assert digest.hexdigest()==row['sha256'] and size==row['bytes']
 if head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[16:18],'little') in (2,3) and int.from_bytes(head[18:20],'little')==183:images.add(row['sha256'])
 verified[p]=identity
assert {p.resolve() for p in (BASE/'objects').glob('*.gz')}=={(BASE/r['artifact']).resolve() for r in rows.values() if r['artifact'].startswith('objects/')}
assert sha((BASE/m['base_manifest']['path']).read_bytes())==m['base_manifest']['sha256']
def data(name):return gzip.decompress((BASE/rows[name]['artifact']).read_bytes())
def js(name):return json.loads(data('journal/'+name+'.json'))
def provider(path):return data('provider/'+path)
old=js('result');new=js('corrected/result');assert old['source_pin']==new['source_pin']==m['source_pin']=='8949313b0a56c0c3bd52509fb363edb4eefba3e4'
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=REPO,stdin=subprocess.PIPE,stdout=subprocess.PIPE);source={}
for row in m['git_sources']:
 proc.stdin.write((m['source_pin']+':'+row['path']+'\n').encode());proc.stdin.flush();head=proc.stdout.readline().split();assert head[1]==b'blob';raw=proc.stdout.read(int(head[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==row['sha256']==rows['source/'+row['path']]['sha256'];source[row['path']]=row['sha256']
proc.stdin.close();assert proc.wait()==0 and len(source)==3520
component_pin='c93263c64f6d5dd1db0e856284d76f1c8e0fa131'
component_names={'src/c_backend.c','src/c_backend.h','src/c_backend_values.inc','src/type_metadata_queries.inc','tests/support/public_c_fixture_providers.c','tests/test_cseed_native_publication.py'} | {name for name in source if name.startswith(('tests/test_source_c','tests/test_public_c_')) and name.endswith('.py')}
for name in sorted(component_names):
 prior=subprocess.check_output(['git','show',component_pin+':'+name],cwd=REPO);assert sha(prior)==source[name]
prior_header=subprocess.check_output(['git','show',component_pin+':src/nanolang.h'],cwd=REPO).splitlines();current_header=data('source/src/nanolang.h').splitlines();assert len(prior_header)==len(current_header)
header_diff=[(a,b) for a,b in zip(prior_header,current_header) if a!=b];assert len(header_diff)==1
assert all(b'char *current_module;' in a and b'char *current_module;' in b and a.split(b'/*',1)[0]==b.split(b'/*',1)[0] for a,b in header_diff)
integrity=js('root-integrity');assert len(integrity)==2 and all(r['head']==m['source_pin'] and r['diff_exit']==0 and r['stdout']==r['stderr']=='' for r in integrity)
def events(folder):
 result=[]
 for line in provider(folder+'/stderr').splitlines():
  try:result.append(json.loads(line))
  except ValueError:pass
 return result
for q in [old,new]:
 assert q['terminal'] and not q['ordinary_ownership_changed'] and q['original_shadows']==1306 and q['outer_timeout']==600 and q['shadow_timeout']=='default10'
 assert q['filesystem'] not in ['nfs','nfs4','cifs']
 assert q['initial_outputs']==dict(obj=False,lib=False,bin=False) and q['before']==source
 assert q['tools_before']==q['tools_after'] and q['providers_before']==q['providers_after']
 for path,digest in q['tools_before'].items():assert sha(provider(path))==digest
 for path,digest in q['providers_before'].items():assert sha(provider(path))==digest
 for health in q['health']:
  assert not health['other_native'] and len(health['rounds'])==3
  assert all(r['cpu']>=1 and r['share']>=.75 and abs(r['share']-r['cpu']/r['wall'])<1e-12 for r in health['rounds'])
 for name,row in q['copied_outputs'].items():
  if name=='bin/nanoc_c':assert sha(provider(q['tree']+'/'+name))==row['sha256']
 for name,row in q['final_outputs'].items():assert sha(provider(q['tree']+'/'+name))==row['sha256'] and len(provider(q['tree']+'/'+name))==row['bytes']
 for s in q['stages']:
  assert s['timeout']==600 and s['shadow_timeout']=='default10' and s['after']==source
  assert len(s['args'])==4 and s['args'][1]==q['tree']+'/src_nano/nanoc_v06.nano' and s['args'][2]=='-o'
  assert s['environment']['NANO_SHADOW_TRACE']==s['environment']['NANO_SHADOW_PHASE_TRACE']=='1'
  assert sha(provider(s['args'][0]))==s['image_sha256']
  es=events(q['work']+'/'+s['name']);starts=[e for e in es if e.get('event')=='shadow-start'];completed=[e for e in es if e.get('event')=='shadow-complete']
  assert len(starts)==len(completed)==s['starts']==s['completions']==1306
  assert [(e['name'],e['source'],e['line']) for e in starts]==[(e['name'],e['source'],e['line']) for e in completed]
  assert {e['phase']:e for e in es if e.get('event')=='cseed-shadow-phase'}['report-returned']['value']==1
assert old['controller_exit']==1 and len(old['stages'])==1
s=old['stages'][0];assert s['exit']==1 and not s['timed_out'] and s['product'] is None
es=events(old['work']+'/ordinary');failed=[e for e in es if e.get('event')=='shadow-complete' and e.get('failures',0)>0];assert len(failed)==1 and failed[0]['name']=='compiler_tool_root' and failed[0]['failures']==2
assert b"Shadow test 'compiler_tool_root' FAILED: 2 failure(s)" in provider(old['work']+'/ordinary/stderr')
status=next(e for e in es if e.get('phase')=='supervisor-status');assert status['exit_code']==1 and not status['completed'] and not status['parent_timeout'] and not status['clock_failure'] and status['signal']==0
assert not any(k in rows for k in ['provider/'+old['tree']+'/bin/nano_vm','provider/'+old['tree']+'/bin/nvm2c', 'provider/'+old['work']+'/ordinary/seed'])
analysis=js('first-failure-analysis');assert analysis['failed_assertions']==2 and analysis['product_absent'] and analysis['failed_shadows']==failed
assert new['controller_exit']==0 and len(new['stages'])==2 and [s['name'] for s in new['stages']]==['ordinary','private']
for s in new['stages']:
 assert s['exit']==0 and not s['timed_out'] and not s['failed_shadows'] and s['help']['exit']==0 and s['help']['timeout']==10
 raw=provider(s['args'][3]);assert sha(raw)==s['product']['sha256'] and len(raw)==s['product']['bytes'] and s['product']['sha256'] in images
 es=events(new['work']+'/'+s['name']);assert all(e['failures']==0 for e in es if e.get('event')=='shadow-complete')
 phases={e['phase']:e for e in es if e.get('event')=='cseed-shadow-phase'}
 for name in ['dispatch-returned','completion-written']:assert phases[name]['value']==1
 assert {'report-start','report-returned','dispatch-start','dispatch-returned','completion-written','flush-start','flush-end','supervisor-status'}<=set(phases)
 status=phases['supervisor-status'];assert status['exit_code']==0 and status['completed'] and not status['parent_timeout'] and not status['clock_failure'] and status['signal']==0 and status['reaped'] and status['deadline_seconds']==10
assert new['missing_companions_before_build']==dict(nano_vm=False,nvm2c=False)
for name,row in new['companions'].items():assert sha(provider(new['tree']+'/bin/'+name))==row['sha256'] and row['sha256'] in images
build=next(r for r in new['setup'] if r['label']=='companion-build');assert build['exit']==0 and build['args'][-3:]==['bin/nano_vm','bin/nvm2c','bin/nanoisa']
assert '-j1' in build['args'] and 'CC=/usr/bin/cc' in build['args']
assert b'/usr/bin/cc ' in provider(new['work']+'/companion-build.stdout')
tool=new['tool_controls'];assert tool==dict(tests=1,failures=0,errors=0,skips=0,calls=20)
commands=json.loads(provider(new['work']+'/tool-cases/calls.json'));assert len(commands)==20 and all(r['exit']==0 for r in commands)
assert sum(r['args'][0]==new['tree']+'/bin/nano_vm' for r in commands)==4
assert sum(r['args'][0]==new['tree']+'/bin/nvm2c' for r in commands)==4
assert sum(r['args'][0]==new['tree']+'/bin/nanoisa' for r in commands)==4
fixture=provider(new['work']+'/focus.nano');extract=new['focused_extraction'];original=data('source/src_nano/nanoc_v06.nano');assert sha(original)==extract['source_sha256'] and sha(fixture)==extract['fixture_sha256']
assert len(extract['ranges'])==15
for item in extract['ranges']:
 raw=original.decode()[item['start']:item['end']].encode();assert sha(raw)==item['sha256'] and raw in fixture
assert len([i for i in extract['ranges'] if i['name']=='compiler_tool_root'])==2
focus=provider(new['work']+'/focused-original-boundary.stderr');fe=[]
for line in focus.splitlines():
 try:fe.append(json.loads(line))
 except ValueError:pass
assert len([e for e in fe if e.get('event')=='shadow-complete'])==9 and all(e['failures']==0 for e in fe if e.get('event')=='shadow-complete')
assert sha(provider(new['work']+'/focus-native'))==new['focused_product']['sha256']
for label in ['original-tool-controls','focused-original-boundary','focus-native','private-link']:assert next(r for r in new['setup'] if r['label']==label)['exit']==0
pub=js('native-publication');assert pub['before']==pub['after'] and len(pub['profiles'])==2
for p in pub['profiles']:
 assert p['exit']==0 and (p['result']['tests'],p['result']['failures'],p['result']['errors'],p['result']['skips'],p['result']['calls'])==(18,0,0,0,32)
 assert provider(pub['work']+'/'+p['mode']+'/tests.log').endswith(b'\nOK\n')
 assert p['compiler']==next(s for s in new['stages'] if s['name']==p['mode'])['args'][0]
 calls=json.loads(provider(pub['work']+'/'+p['mode']+'/calls.json'));assert len(calls)==32 and all(r.get('timeout') in [30,120] for r in calls)
for path,digest in pub['after'].items():assert sha(provider(path))==digest
current_public=js('public-c-controls');assert len(current_public['modules'])==17
assert (current_public['tests'],current_public['failures'],current_public['errors'],current_public['skips'])==(66,0,0,0)
assert len(current_public['commands'])==685 and current_public['commands']==js('public-c-commands')
assert current_public['image']==new['stages'][0]['args'][0] and current_public['image_sha256']==new['stages'][0]['image_sha256']
assert b'Ran 66 tests' in provider(current_public['work']+'/tests.log') and provider(current_public['work']+'/tests.log').endswith(b'\nOK\n')
assert len([name for name in component_names if name.startswith('tests/test_public_c_')])==17
enum=js('enum-profile');assert enum['source_pin']==m['source_pin'] and len(enum['commands'])==38
assert enum['tools_before']==enum['tools_after']
for path,digest in enum['tools_before'].items():assert sha(provider(path))==digest
assert provider(enum['work']+'/original.nano')==data('source/tests/nanoisa/fixtures/enum_int64_values.nano')
assert len([r for r in enum['commands'] if r['label'].endswith('-build')])==16 and len([r for r in enum['commands'] if r['label'].endswith('-run')])==16
for r in enum['commands']:
 expected=1 if r['label'].endswith('-unknown') else 0;assert r['exit']==expected and r['timeout'] in [30,120]
 if r['label'].endswith('-build'):
  assert {'-Wall','-Wextra','-Werror'}<=set(r['args']) and any(s in r['args'] for s in ['-std=c99','-std=c11'])
for mode in ['ordinary','private']:
 assert provider(enum['work']+'/'+mode+'-previous.c')==b'I retain prior enum C output.\n'
 assert b'Absent' in provider(enum['work']+'/'+mode+'-unknown.stderr')
mapped=js('corrected/loaded-images');assert mapped
continuity=js('runtime-library-continuity');assert len(continuity['observations'])==len(mapped)
for record in mapped:
 for path,digest in record['files'].items():assert sha(provider(path))==digest
for row in continuity['observations']:assert len(row['libraries'])==5 and len(row['new_image'])==1
assert {k for k,v in m['acceptance'].items() if v}=={'ordinary_reference_compiler_source','private_reference_compiler_source','companion_tool_boundary','current_native_publication','current_public_c_profile','current_enum_namespace'}
print(json.dumps(dict(result='PASS',source=m['source_pin'],records=len(rows),git_inputs=len(source),native_images=len(images),original_shadows_per_mode=1306,ordinary_pass=True,private_pass=True,retained_setup_assertion_failures=2,tool_native_cases=4,tool_commands=20,focused_original_shadows=9,current_publication_methods_per_mode=18,current_public_c_methods=66,current_public_c_commands=685,enum_namespace_commands=38,unchanged_profile_components=len(component_names),ordinary_ownership_enabled=False,raw_bootstrap=False,paired_platform=False,whole_release=False),indent=2))
