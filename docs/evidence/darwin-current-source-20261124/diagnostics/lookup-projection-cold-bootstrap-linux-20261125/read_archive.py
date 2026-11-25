"""I verify the exact fresh bootstrap and its original native-consumer terminal."""
from pathlib import Path
import sys,json,hashlib,subprocess,re,ast
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'eae3850a91bf73dd8e1dc5fac790392f8aaaa7fc63ded80d02180bd7a56e3c2d');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('bootstrap/setup-result.json');p=js('bootstrap/prepare-result.json');b=js('bootstrap/bootstrap-result.json');n=js('bootstrap/native-gates-result.json');sequence=js('bootstrap/sequence-result.json');installed=js('bootstrap/source/bin/nanoc_bootstrap.json')
assert all(r['terminal'] for r in [s,p,b,n,sequence]) and s['admitted_setup'] and p['accepted'] and b['original_make_pass'] and b['exit']==0
assert m['cold_tools'] and m['original_make_pass'] and m['original_native36_pass'] and m['original_emitter12_pass']==n['accepted']==m['native_acceptance'] and m['controller_propagates_failure']
assert not m['whole_release'] and not m['pure_translator'] and not m['paired']
assert (m['stage_seconds'],m['compiler_suite_seconds'],m['ordinary_seconds'],m['helper_command_seconds'])==(1800,30,10,300)
assert s['cold_tools'] and s['provider_copies']=={} and s['source_pin']==b['source_pin']==n['source_pin']==m['source_commit']=='8806139ca8bf3f60a25d88d241f67747c6ef9e8a'
assert s['source_before']==s['source_after']==p['source_before']==p['source_after']==b['source_before']==b['source_after']==n['source_before']==n['source_after']
assert b['images_before']==b['images_after'] and n['images_before']==n['images_after'] and s['tools']==s['tools_after']==p['tools_after']==b['tools_before']==b['tools_after']==n['tools_before']==n['tools_after']
assert b['source_stable'] and b['images_stable'] and b['tools_stable']
for name,h in b['source_before'].items():assert rows['bootstrap/source/'+name]['sha256']==h
for name,h in b['images_after'].items():assert rows['bootstrap/source/'+name]['sha256']==h
for name,h in s['tools'].items():assert rows['external/'+name.lstrip('/')]['sha256']==h
for name,h in installed['hosts'].items():
 assert rows['external/'+name.lstrip('/')]['sha256']==h
 if Path(name).is_relative_to(Path(m['roots']['bootstrap'])):assert rows['bootstrap/'+str(Path(name).relative_to(m['roots']['bootstrap']))]['sha256']==h
assert all(c['exit']==0 and not c.get('timed_out') for c in s['commands']+p['commands'])
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
x=data('bootstrap/source/bin/nanoc_stage1.nvm');y=data('bootstrap/source/bin/nanoc_stage2.nvm');assert x==y and len(x)==1059564 and sha(x)=='1e57ed3b7fcabba4137c9c6592d985d46745e079c3dc7f53a675fd067bc762d2'
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
assert all(row['exit']==0 for row in sequence['phases'][:-1])
assert sequence['phases'][-1]['script']=='run_native_gates.py' and sequence['phases'][-1]['exit']==(0 if n['accepted'] else 1)
assert b"if not q.get('accepted'):raise SystemExit(1)" in data('bootstrap/run_native_gates.py')
for name,h in n['images_after'].items():
 if name=='bin/nanoc':assert rows['bootstrap/source/bin/nanoc_stage2']['sha256']==h
 else:assert rows['bootstrap/source/'+name]['sha256']==h
observation=js('live-observation/observation.json');assert observation['read_only'] and not observation['terminal_observation'] and not observation['product_acceptance'] and observation['source_pin']==m['source_commit']
assert observation['bootstrap_result_sha256']==rows['bootstrap/bootstrap-result.json']['sha256'] and observation['bootstrap_receipt_sha256']==rows['bootstrap/source/bin/nanoc_bootstrap.json']['sha256']
assert observation['producer_sha256']==rows['bootstrap/source/bin/nanoc_stage2']['sha256'] and observation['input_sha256']==rows['bootstrap/source/tests/selfhost_shadow_emitter.nano']['sha256']
assert observation['argv'][0]==str(Path(m['roots']['bootstrap'])/'source/bin/nanoc_stage2') and observation['argv'][1]==str(Path(m['roots']['bootstrap'])/'source/tests/selfhost_shadow_emitter.nano')
assert observation['argv']==[v.decode() for v in data('live-observation/proc-cmdline').split(b'\0') if v] and len(observation['files'])==1
for file in observation['files']:
 assert file['stable_stat'] and file['bytes']==rows['live-observation/'+file['snapshot']]['bytes'] and file['sha256']==rows['live-observation/'+file['snapshot']]['sha256'] and data('live-observation/'+file['snapshot'])[:4]==b'\x7fELF'
 assert Path(file['path']).parent==Path(observation['argv'][-1]).parent and Path(file['path']).name=='nanoc_c'
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in b['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',name;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
defined_methods=[v for v in ast.walk(ast.parse(data('bootstrap/source/tests/test_native_shadow_emitter.py'))) if isinstance(v,ast.FunctionDef) and v.name.startswith('test_')];assert len(defined_methods)==12
proc.stdin.close();assert proc.wait()==0
oldm,oldrows,olddata,oldjs=read_archive(base.parent/'shadow-helper-cold-bootstrap-linux-20261124','c8c6cfe80993b0610a07576d13338b282a402d87fa6c636231d7680f923300e5')
previous_setup=oldjs('bootstrap/setup-result.json');assert s['flags']==previous_setup['flags']
assert s['make']==['/usr/bin/make','-j1','-f','Makefile.gnu','--no-print-directory','CC=/usr/bin/cc','CFLAGS='+' '.join(s['flags'])]
for name in ['tests/test_native_shadows.py','tests/test_shadow_runner.py','tests/test_make_timeouts.py','tests/test_native_shadow_only_entry.py','tests/test_native_shadow_helpers.py','tests/test_native_shadow_emitter.py','tests/selfhost_shadow_emitter.nano']:
 assert rows['bootstrap/source/'+name]['sha256']==oldrows['bootstrap/source/'+name]['sha256'],name
for target in [b'test-native-shadows',b'test-native-shadow-emitter']:
 recipe=lambda raw:re.search(rb'(?m)^'+re.escape(target)+rb':[^\n]*\n(?:\t[^\n]*(?:\n|$))+',raw).group(0)
 assert recipe(data('bootstrap/source/Makefile.gnu'))==recipe(olddata('bootstrap/source/Makefile.gnu'))
print('PASS %d records: original fresh own-Stage2 bootstrap/all17/raw%d equality/install/no-C-seed, original native36 PASS, original emitter%d exit%d/native_acceptance=%s; stable source/tool/image/host and unchanged original tests/recipes. No paired/current release, full pure-translator or whole-runtime sanitizer claim.'%(len(rows),len(x),methods,z['exit'],n['accepted']))
