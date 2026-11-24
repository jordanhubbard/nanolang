"""I independently verify cold-bootstrap/native results and exact artifact observations."""
from pathlib import Path
import sys,json,hashlib,subprocess,re,ast
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'b3c85f73568c2c21784de9df63b80b68f9df1e296abdafab91e983de3559ae15');sha=lambda raw:hashlib.sha256(raw).hexdigest()
b=js('bootstrap/bootstrap-result.json');setup=js('bootstrap/setup-result.json');prep=js('bootstrap/prepare-result.json');dependent=js('bootstrap/dependent-native-controller.json');a=js('artifact-diagnostic/result.json')
assert b['terminal'] and setup['terminal'] and prep['terminal'] and dependent['terminal'] and a['terminal']
assert setup['admitted_setup'] and setup['cold_tools'] and setup['provider_copies']=={} and prep['accepted']
assert setup['source_pin']==b['source_pin']==m['source_commit']=='16b081fdf472dfe233ebf5975f149f6aa9d33783'
assert m['cold_tools'] and not m['copied_producer'] and not m['artifact_production_admission'] and not m['shared_library_semantic_equivalence'] and not m['whole_release'] and not m['complete_san_acceptance']
assert (m['stage_seconds'],m['compiler_suite_seconds'],m['ordinary_shadow_seconds'])==(1800,30,10)
assert b['source_stable'] and b['images_stable'] and b['tools_stable']
assert b['source_before']==b['source_after']==setup['source_before']==setup['source_after']==prep['source_before']==prep['source_after']
assert b['images_before']==b['images_after'] and b['tools_before']==b['tools_after']==prep['tools_after']==setup['tools']
for n,h in b['source_before'].items():assert rows['bootstrap/source/'+n]['sha256']==h,n
for n,h in b['images_before'].items():assert rows['bootstrap/source/'+n]['sha256']==h,n
for n,h in b['tools_before'].items():assert rows['tool/'+n.lstrip('/')]['sha256']==h,n
for c in setup['commands']:assert c['exit']==0 and not c.get('timed_out',False)
assert len([c for c in setup['commands'] if c['label']=='cold-tools'])==1
assert b['exit']==m['original_make_exit'] and bool(b.get('original_make_pass'))==m['original_make_pass']
manifest_paths=[n for n in rows if n.startswith('bootstrap/source/obj/bootstrap-nanoisa/run-') and n.endswith('/manifest.json')];assert len(manifest_paths)==1;receipt=js(manifest_paths[0]);prefix=manifest_paths[0].rsplit('/',1)[0]+'/'
for step in receipt['steps']:
 name=Path(step['log']).name
 if 'log_sha256' in step:assert rows[prefix+name]['sha256']==step['log_sha256']
if b.get('original_make_pass'):
 assert b['exit']==0 and len(receipt['steps'])==17 and all(c['exit_code']==0 for c in receipt['steps'])
 installed=js('bootstrap/source/bin/nanoc_bootstrap.json');assert installed['phase']=='stage2' and installed['raw_stage1_stage2_equal']
 assert installed['stage_timeout_seconds']==1800 and installed['shadow_timeout_seconds']==30
 expected=['seed','seed-verify','stage1','stage1-verify','stage1-translate','stage1-native','stage1-hello-compile','stage1-hello-execute','stage2','stage2-verify','stage2-translate','stage2-native','stage2-hello-compile','stage2-hello-execute','seed-final-verify','stage1-final-verify','stage2-final-verify']
 assert [v['label'] for v in installed['steps']]==expected
 journal=Path(m['roots']['bootstrap'])
 def archived(path):return 'bootstrap/'+str(Path(path).relative_to(journal))
 for step in installed['steps']:
  assert step['exit_code']==0 and not step.get('interrupted_or_timed_out',False) and step['elapsed_seconds']<1800
  assert rows[archived(step['log'])]['sha256']==step['log_sha256']
 for group in ['artifacts','tools']:
  for name,h in installed[group].items():assert rows['bootstrap/source/bin/'+name]['sha256']==h
 for name,h in installed['sources'].items():assert rows['bootstrap/source/'+name]['sha256']==h
 for path,h in installed['hosts'].items():assert rows[archived(path)]['sha256']==h
 assert not any(name.endswith('/unexpected-native-compiler') for name in rows)
 out=data('bootstrap/bootstrap.stdout');assert b'I installed bin/nanoc from the verified Stage 2 module.' in out and b'bin/nanoc succeeded with bin/nanoc_c removed' in out

 x=data('bootstrap/source/bin/nanoc_stage1.nvm');y=data('bootstrap/source/bin/nanoc_stage2.nvm');assert x==y and len(x)>0
 assert all('bootstrap/source/'+n in rows for n in ['.bootstrap1.built','.bootstrap2.built','.bootstrap3.built','.bootstrap0.built'])
 native=js('bootstrap/native-gates-result.json');assert native['terminal'] and native['source_pin']==m['source_commit']
 assert native['source_before']==native['source_after']==b['source_before'] and native['images_before']==native['images_after'] and native['tools_before']==native['tools_after']==b['tools_before']
 commands=native['commands'];assert commands and commands[0]['target']=='test-native-shadows'
 summary=data('bootstrap/test-native-shadows.stderr');assert re.search(rb'Ran 24 tests? in ',summary)
 assert all(c['observation_timeout_seconds']==1800 for c in commands)
 if commands[0]['exit']==0:assert len(commands)==2 and commands[1]['target']=='test-native-shadow-emitter' and re.search(rb'Ran 12 tests? in ',data('bootstrap/test-native-shadow-emitter.stderr'))
 else:assert len(commands)==1 and not m['emitter_companion_executed']
 assert m['original_native_gates']==native.get('accepted',False)==dependent.get('accepted',False)
 assert bool(native.get('accepted',False))==(len(commands)==2 and all(c['exit']==0 and not c.get('outer_timeout',False) for c in commands))
 for n,h in native['images_after'].items():
  if n=='bin/nanoc':
   links={v['path']:v['target'] for v in m['symlinks']};assert links['bootstrap/source/bin/nanoc']=='nanoc_stage2' and rows['bootstrap/source/bin/nanoc_stage2']['sha256']==h
  else:assert rows['bootstrap/source/'+n]['sha256']==h
else:
 assert not dependent.get('accepted',False) and not m['original_native_gates'] and not m['native_gates_executed']
assert a['diagnostic_only'] and not a['production_admission'] and not a['whole_release'] and not a['shared_library_semantic_equivalence']
assert a['source_pin']==m['artifact_source_commit']=='cbbfdcb70cb192abaf3f043fe07b7c8e7fa99db2'
parent=base.parent/'core-supervisor-bootstrap-linux-20261124';pm,pr,pdata,pjs=read_archive(parent,'dc9080dac206655d64267f2e3755fcd463376d83ffdc35d359225593c36421db');pb=pjs('bootstrap/bootstrap-result.json')
if 'source_before' in a:
 assert a['source_before']==a['source_after']==pb['source_before'] and a['tools_before']==a['tools_after']
 for n,h in a['tools_before'].items():
  assert rows['tool/'+n.lstrip('/')]['sha256']==h
  if '/source/' in n:assert pr['bootstrap/source/'+n.split('/source/',1)[1]]['sha256']==h
 assert a['tools_before']['/usr/bin/cc']==b['tools_before']['/usr/bin/cc']
if a.get('accepted_diagnostic'):
 assert m['artifact_diagnostic_accepted'] and not a.get('exception')
 commands={c['label']:c for c in a['commands']};assert len(commands)==len(a['commands'])
 for name in ['marker','missing-symbol','missing-artifact','cancel','early']:
  assert commands[name+'-assemble']['exit']==0 and commands[name+'-verify']['exit']==0
  check=commands[name+'-check'];assert check['timeout']==90 and not check.get('timed_out',False);assert check['exit']==0 if name=='marker' else check['exit']!=0
  assert commands[name+'-c-refusal']['exit']!=0 and data('artifact-diagnostic/'+name+'.c-output')==b'prior output'
  stderr=data('artifact-diagnostic/'+name+'-check.stderr')
  if name=='cancel':assert b'after 10 seconds' in stderr
  if name=='early':assert b'shadow' in stderr
 assert data('artifact-diagnostic/marker.marker')==b'native'
 for name in ['cancel','early']:
  pm2=json.loads((base.parent/'native-shadow-localization-linux-20261124/manifest.json').read_bytes());rr={r['path']:r for r in pm2['files']};i=0 if name=='cancel' else 1;assert rows['artifact-diagnostic/'+name+'.c']['sha256']==rr['localization/processes/%03d/before/'%i+name+'.c']['sha256']
 for n,v in a['native_products'].items():assert rows['artifact-diagnostic/'+n]['sha256']==v['sha256']
else:assert not m['artifact_diagnostic_accepted'] and a.get('exception')
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(pin,name):
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',name;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';return raw
for pin,guard in [(m['source_commit'],b['source_before']),(m['artifact_source_commit'],a.get('source_before',{}))]:
 for n,h in guard.items():assert sha(blob(pin,n))==h,(pin,n)
old=ast.parse(blob(m['artifact_source_commit'],'tests/test_native_shadows.py'));new=ast.parse(data('bootstrap/source/tests/test_native_shadows.py'))
def methods(tree):return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
assert methods(old)==methods(new)
proc.stdin.close();assert proc.wait()==0
print('PASS integrity: %d records; original Make exit=%s, native acceptance=%s, bounded artifact observation=%s. No complete sanitizer, paired, pure-translator or release acceptance.'%(len(rows),b.get('exit'),m['original_native_gates'],m['artifact_diagnostic_accepted']))
