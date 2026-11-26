"""I verify the exact fresh bootstrap and its original native-consumer terminal."""
from pathlib import Path
import sys,json,hashlib,subprocess,re,ast
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'d230b2bd730561baa99e3496e8c58eb14fd570e24176c90bba9cb255f9df9f8b');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('bootstrap/setup-result.json');p=js('bootstrap/prepare-result.json');b=js('bootstrap/bootstrap-result.json');n=js('bootstrap/native-gates-result.json');installed=js('bootstrap/source/bin/nanoc_bootstrap.json')
assert all(r['terminal'] for r in [s,p,b,n]) and s['admitted_setup'] and p['accepted'] and b['original_make_pass'] and b['exit']==0
refusal=js('collector-source/live-refusal-result.json')
assert refusal['terminal'] and refusal['accepted'] and refusal['exit']==1 and not refusal['archive_created']
assert refusal['live_result_sha256']==rows['collector-source/live-bootstrap-result.json']['sha256'] and not js('collector-source/live-bootstrap-result.json')['terminal']
assert refusal['collector_sha256']==rows['collector-source/package.py']['sha256'] and b'AssertionError' in data('collector-source/live-refusal.stderr')
native_refusal=js('collector-source/live-native-refusal-result.json')
assert native_refusal['terminal'] and native_refusal['accepted'] and native_refusal['exit']==1 and not native_refusal['archive_created']
assert native_refusal['live_result_sha256']==rows['collector-source/live-native-result.json']['sha256'] and not js('collector-source/live-native-result.json')['terminal']
assert native_refusal['collector_sha256']==rows['collector-source/package.py']['sha256'] and b'AssertionError' in data('collector-source/live-native-refusal.stderr')
assert m['cold_tools'] and m['original_make_pass'] and m['original_native36_pass'] and m['original_emitter12_pass']==n['accepted']==m['native_acceptance']
assert not m['whole_release'] and not m['pure_translator'] and not m['paired']
assert (m['stage_seconds'],m['compiler_suite_seconds'],m['ordinary_seconds'],m['helper_command_seconds'])==(1800,30,10,300)
assert s['cold_tools'] and s['provider_copies']=={} and s['source_pin']==b['source_pin']==n['source_pin']==m['source_commit']=='771a14330475bcd784d7cb19a215d6a4f15728d4'
assert s['source_before']==s['source_after']==p['source_after']==b['source_before']==b['source_after']==n['source_before']==n['source_after']
assert b['images_before']==b['images_after'] and n['images_before']==n['images_after'] and s['tools']==s['tools_after']==p['tools_after']==b['tools_before']==b['tools_after']==n['tools_before']==n['tools_after']
assert b['source_stable'] and b['images_stable'] and b['tools_stable']
for name,h in b['source_before'].items():assert rows['bootstrap/source/'+name]['sha256']==h
for name,h in b['images_after'].items():assert rows['bootstrap/source/'+name]['sha256']==h
for name,h in s['tools'].items():assert rows['external/'+name.lstrip('/')]['sha256']==h
for name,h in installed['hosts'].items():
 assert rows['external/'+name.lstrip('/')]['sha256']==h
 if Path(name).is_relative_to(Path(m['roots']['bootstrap'])):assert rows['bootstrap/'+str(Path(name).relative_to(m['roots']['bootstrap']))]['sha256']==h
assert all(c['exit']==0 and not c.get('timed_out') for c in s['commands'])
assert len([c for c in s['commands'] if c['label']=='cold-tools'])==1
assert installed['phase']=='stage2' and installed['raw_stage1_stage2_equal'] and installed['stage_timeout_seconds']==1800 and installed['shadow_timeout_seconds']==30
labels=['seed','seed-verify','stage1','stage1-verify','stage1-translate','stage1-native','stage1-hello-compile','stage1-hello-execute','stage2','stage2-verify','stage2-translate','stage2-native','stage2-hello-compile','stage2-hello-execute','seed-final-verify','stage1-final-verify','stage2-final-verify']
assert [x['label'] for x in installed['steps']]==labels
for x in installed['steps']:
 assert x['exit_code']==0 and not x.get('interrupted_or_timed_out',False) and x['elapsed_seconds']<1800
 key='bootstrap/'+str(Path(x['log']).relative_to(m['roots']['bootstrap']));assert rows[key]['sha256']==x['log_sha256']
for group in ['artifacts','tools']:
 for name,h in installed[group].items():assert rows['bootstrap/source/bin/'+name]['sha256']==h
for name,h in installed['sources'].items():assert rows['bootstrap/source/'+name]['sha256']==h
x=data('bootstrap/source/bin/nanoc_stage1.nvm');y=data('bootstrap/source/bin/nanoc_stage2.nvm');assert x==y and len(x)==1060388 and sha(x)=='bfc22c0b878c8f7a25486a7bc58e6792cd901649892595d1db1c6be93c2c1944'
assert all('bootstrap/source/.bootstrap%d.built'%i in rows for i in range(4))
links={r['path']:r['target'] for r in m['symlinks']};assert links['bootstrap/source/bin/nanoc']=='nanoc_stage2'
bootlog=data('bootstrap/bootstrap.stdout');assert b'I installed bin/nanoc from the verified Stage 2 module.' in bootlog and b'bin/nanoc succeeded with bin/nanoc_c removed' in bootlog
assert [c['target'] for c in n['commands']]==['test-native-shadows','test-native-shadow-emitter']
a,z=n['commands'];assert a['exit']==0 and all(c['observation_timeout_seconds']==1800 and not c.get('outer_timeout') for c in n['commands'])
assert re.search(rb'Ran 36 tests? in ',data('bootstrap/test-native-shadows.stderr')) and data('bootstrap/test-native-shadows.stderr').rstrip().endswith(b'OK')
emitter=data('bootstrap/test-native-shadow-emitter.stderr');counts=re.findall(rb'Ran ([0-9]+) tests?',emitter);assert len(counts)==1;methods=int(counts[0])
if n['accepted']:
 assert z['exit']==0 and methods==12 and emitter.rstrip().endswith(b'OK') and b'skipped=' not in emitter
else:
 assert z['exit']!=0 and b'FAILED (' in emitter
if not n['accepted']:
 assert z['exit']!=0 and b'FAILED (' in emitter
 # I preserve the actual terminal, not a required historical failure.
 if b'timed out after 300 seconds' in emitter:assert methods==0 and b'FAILED (errors=1)' in emitter
assert b"if not q.get('accepted'):raise SystemExit(1)" in data('bootstrap/run_native_gates.py')
for name,h in n['images_after'].items():
 if name=='bin/nanoc':assert rows['bootstrap/source/bin/nanoc_stage2']['sha256']==h
 else:assert rows['bootstrap/source/'+name]['sha256']==h
observation=js('bootstrap/raw-equality-observation.json');assert observation['raw_equal'] and observation['bytes']==len(x) and observation['sha256']==sha(x)
assert installed['artifacts']['nanoc_stage2']==rows['bootstrap/source/bin/nanoc_stage2']['sha256']
parent_m,parent_rows,parent_data,parent_js=read_archive(base.parent/'emitter-record-list-lookup-five-linux-20261126','6ffc195991c8a6dbef1ef9ab53a897887896f79a98fc2542d797d4d7d1d1fdf5')
component=parent_js('qualified/qualification-result.json');material=parent_js('collector-source/materialization.json')
assert component['terminal'] and component['accepted_component'] and component['all_guards_match'] and len(component['commands'])==270
assert material['terminal'] and material['accepted_call_path_observation']
for file,key in [('component.json','qualified/qualification-result.json'),('materialization.json','collector-source/materialization.json')]:assert data('bootstrap/receipts/'+file)==parent_data(key)
for file,h in component['roles']['candidate']['source_after'].items():
 if file.startswith(('src/','modules/','stdlib/','src_nano/')) and not file.startswith('src_nano/nvm2c/'):assert rows['bootstrap/source/'+file]['sha256']==h,file
assert s['component_publication']==m['source_commit'] and s['component_manifest_sha256']=='6ffc195991c8a6dbef1ef9ab53a897887896f79a98fc2542d797d4d7d1d1fdf5'
assert p['admission_only'] and not p['complete_san_acceptance'] and p['known_san_failure']==1252
assert p['full_five_routes'] and p['permanent_unittest_invoked'] is False
for name,h in p['prerequisite_receipts'].items():assert rows['prerequisites/'+Path(name).parent.name+'/'+Path(name).name]['sha256']==h
audit=js('bootstrap/post-terminal-audit.json');assert audit['terminal'] and audit['source_match'] and audit['native_guards_match'] and not audit['other_jobs'] and audit['tools_match'] and audit['hosts_match']
assert not p['other_jobs'] and p['free_bytes']>15*1024**3
assert all(v['cpu']/v['wall']>=.75 and v['cpu']>=1 for v in p['cpu_rounds'])
assert not s['working_overlay'] and not s['overlays']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in b['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',name;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
defined_methods=[v for v in ast.walk(ast.parse(data('bootstrap/source/tests/test_native_shadow_emitter.py'))) if isinstance(v,ast.FunctionDef) and v.name.startswith('test_')];assert len(defined_methods)==12
proc.stdin.close();assert proc.wait()==0
oldm,oldrows,olddata,oldjs=read_archive(base.parent/'lookup-hoisted-cold-bootstrap-linux-20261125','d232ed852a6f375dc76f1d34cac02b22c87aa05c5b220073180f8dbbe39d7e49')
previous_setup=oldjs('bootstrap/setup-result.json');assert s['flags']==previous_setup['flags']
assert s['make']==['/usr/bin/make','-j1','-f','Makefile.gnu','--no-print-directory','CC=/usr/bin/cc','CFLAGS='+' '.join(s['flags'])]
for name in ['tests/test_native_shadows.py','tests/test_shadow_runner.py','tests/test_make_timeouts.py','tests/test_native_shadow_only_entry.py','tests/test_native_shadow_helpers.py','tests/test_native_shadow_emitter.py','tests/selfhost_shadow_emitter.nano']:
 assert rows['bootstrap/source/'+name]['sha256']==oldrows['bootstrap/source/'+name]['sha256'],name
for target in [b'test-native-shadows',b'test-native-shadow-emitter']:
 recipe=lambda raw:re.search(rb'(?m)^'+re.escape(target)+rb':[^\n]*\n(?:\t[^\n]*(?:\n|$))+',raw).group(0)
 assert recipe(data('bootstrap/source/Makefile.gnu'))==recipe(olddata('bootstrap/source/Makefile.gnu'))

native_admission=js('bootstrap/native-consumer-admission.json')
assert native_admission['terminal'] and native_admission['accepted'] and native_admission['admission_only'] and not native_admission['native_acceptance'] and not native_admission['whole_release']
assert native_admission['source_pin']==m['source_commit'] and native_admission['raw_equal'] and native_admission['raw_bytes']==len(x) and native_admission['raw_sha256']==sha(x)
assert native_admission['stage2_sha256']==installed['artifacts']['nanoc_stage2']
assert native_admission['source_after']==b['source_after'] and native_admission['images_after']==n['images_before'] and native_admission['tools_after']==n['tools_before'] and native_admission['hosts_after']==installed['hosts']
assert native_admission['filesystem']!='nfs' and native_admission['free_bytes']>15*1024**3 and not native_admission['other_jobs'] and all(v['cpu']>=1 and v['share']>=.75 for v in native_admission['cpu_rounds'])
publication_refusal=js('bootstrap/publication-refusal/result.json')
assert publication_refusal['terminal'] and publication_refusal['accepted'] and publication_refusal['accepted_parent'] and publication_refusal['exit']==1
assert publication_refusal['unpublished_commit']==m['source_commit'] and publication_refusal['cold_roots_before']==publication_refusal['cold_roots_after'] and publication_refusal['product_commands']==0 and not publication_refusal['source_created']
assert publication_refusal['parent_sha256']==rows['bootstrap/publication-refusal/parent.json']['sha256']==parent_rows['qualified/qualification-result.json']['sha256']
assert publication_refusal['generator_sha256']==rows['bootstrap/prepare-generator.py']['sha256']==rows['bootstrap/publication-refusal/prepare_bootstrap.py']['sha256']
assert b'ls-remote' in data('bootstrap/publication-refusal/stderr') and b'AssertionError' in data('bootstrap/publication-refusal/stderr')

observer_failure=js('collector-source/observer-first-failure.json')
assert observer_failure['terminal'] and observer_failure['exit']==1 and observer_failure['product_commands']==observer_failure['archive_attempts']==0 and observer_failure['product_tests_unchanged']
assert data('collector-source/capture_native_refusal.py')==data('collector-source/capture_native_refusal_v1.py')
assert b'(satisfied #f)' in data('collector-source/native-refusal-wait.stdout') and b'(reason timeout)' in data('collector-source/native-refusal-wait.stdout')
live_native=js('collector-source/live-native-result.json');assert live_native['source_pin']==n['source_pin'] and live_native['source_before']==n['source_before'] and live_native['images_before']==n['images_before'] and live_native['tools_before']==n['tools_before']
print('PASS %d records: original fresh own-Stage2 bootstrap/all17/raw%d equality/install/no-C-seed, original native36 PASS, original emitter%d exit%d/native_acceptance=%s; stable source/tool/image/host and unchanged original tests/recipes. No paired/current release, full pure-translator or whole-runtime sanitizer claim.'%(len(rows),len(x),methods,z['exit'],n['accepted']))
