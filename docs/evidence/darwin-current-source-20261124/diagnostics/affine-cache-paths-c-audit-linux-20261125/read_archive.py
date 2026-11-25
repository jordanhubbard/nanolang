"""I verify C cache-path expectations and twelve selected VM controls without claiming a NanoLang route."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'bf5d81efec09172ce4ac9c530d85064ee2a549a5363be2e728a0d2114d27ce9d')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('comparison/result.json')
assert m['source_commit']==q['source_pin']=='884887793345685872a206edf3c49c97ceb6834c'
assert m['component_only'] and m['selected_vm_only'] and not m['c_oracle_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release'])
assert q['terminal'] and q['accepted_c_oracle'] and q['all_guards_match'] and len(q['commands'])==281
def mapped(name):
 p=Path(name)
 for label,root in sorted(m['roots'].items(),key=lambda pair:len(pair[1]),reverse=True):
  if p.is_relative_to(root):return label+'/'+str(p.relative_to(root))
 return 'external/'+str(p).lstrip('/')
for group in ['source','production_source','tools','objects']:
 assert q[group+'_before']==q[group+'_after']
 for name,h in q[group+'_before'].items():assert rows[mapped(name)]['sha256']==h
build=q['commands'][0]['argv'];assert build[0]=='/usr/bin/cc'
for flag in ['-Wall','-Wextra','-Werror','-std=c99','-O3','-fPIC']:assert flag in build
assert rows['origin/bootstrap-result.json']['sha256']==q['origin_bootstrap_sha256']
origin=js('origin/bootstrap-result.json');assert origin['terminal'] and origin['original_make_pass'] and origin['source_pin']=='7b9f08ee0941c23810d18fa3f8ebfbb759c968de'
for i,c in enumerate(q['commands'],1):assert c['exit']==0 and c['timeout'] in [60,600] and data('comparison/command-%04d.stderr'%i)==b''
for group,count,h in [('original',1001,'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'),('cache',3317,'2a6203cea85a5303b9907da8f082a9201717252edaa0516c3b435f7bf217d53c')]:
 corpus=js('comparison/'+group+'-manifest.json');assert len(corpus)==count
 reports=[]
 for i,c in enumerate(q['commands'][1:],2):
  if not Path(c['argv'][1]).name.startswith(group+'-'):continue
  paths=c['argv'][1:];assert c['timeout']==(600 if group=='original' else 60)
  if group=='original':assert len(paths)<=8
  else:assert len(paths)<=32 and (len(paths)==1 or sum(rows[mapped(p)]['bytes'] for p in paths)<=8192)
  reports.extend(data('comparison/command-%04d.stdout'%i).decode().splitlines())
 raw=('\n'.join(reports)+'\n').encode();assert raw==data('comparison/'+group+'.reports') and len(reports)==count and sha(raw)==h
 assert not next(g for g in q['groups'] if g['group']==group)['failures']
 for case,text in zip(corpus,reports):
  assert rows[mapped(case['path'])]['sha256']==case['sha256'];expected=case['expected']
  if isinstance(expected,str):
   if group=='original' and not case['name'].startswith(('global-algebra-','bounded-fifo-')):assert text.startswith(expected)
   else:assert text==expected
  else:
   valid,message,frame_,authority_,keys=text.split('|',4);local,stack,globals_,context=frame_.split('/')
   assert valid==str(int(expected['valid'])) and message==expected['message'] and stack==expected['stack'],case['name']
   assert globals_=='9,' and context==expected['context'] and authority_==expected['authority'] and keys==expected['keys'],case['name']
parse=js('source-review/result.json');assert parse['terminal'] and parse['all_guards_match']
assert parse['source_before']==parse['source_after'] and parse['tools_before']==parse['tools_after']
assert len(parse['commands'])==5 and all(c['exit']==0 for c in parse['commands'])
assert parse['commands'][-1]['timeout']==600 and parse['commands'][-1]['ordinary_shadow_seconds']==10
for name,h in parse['source_before'].items():
 assert rows['source-review/'+name]['sha256']==h
 if name.endswith(('.nano','.c','.py')):assert rows['comparison/'+name]['sha256']==h
for name,h in parse['tools_before'].items():assert rows[mapped(name)]['sha256']==h
assert origin['images_before']==origin['images_after']
assert not any(s.endswith('/nanoisa/affine_bytecode.o') for s in build)
assert sum(s.endswith('/nanoisa/affine_state.o') for s in build)==1
reference=data('comparison/tests/nvm2c_affine_analysis_cache_reference.c')
assert b'status<1 || status>2' in reference and b'if(!guarded && (caller_present || !graph))' in reference
assert b'analyze(present?&model:NULL,function,caller,0,calls)' in reference
assert b'assert (not entry.proceed)' in data('comparison/tests/nvm2c_affine_analysis_cache_driver.nano')
corpus=js('comparison/cache-manifest.json');assert len(corpus)==3317 and len({c['name'] for c in corpus})==3317
for name in ['fresh-body-outside-protocol','caller-binding-outside-protocol']:
 c=next(c for c in corpus if c['name']==name);assert c['expected']=='D'
micro=js('selected-vm/result.json');assert micro['terminal'] and micro['accepted_micro'] and micro['all_guards_match']
assert micro['source_pin']==m['source_commit'] and len(micro['commands'])==12
assert micro['tool_before']==micro['tool_after']==origin['images_after']['bin/nano_vm']
assert micro['artifact_before']==micro['artifact_after']==rows['selected-vm/driver.nvm']['sha256']==rows['source-review/shadow-checked.nvm']['sha256']
full=data('comparison/cache.reports').decode().splitlines()
for c in micro['commands']:
 assert c['exit']==0 and c['timeout']==60
 case=corpus[c['case_index']];assert case['name']==c['case'] and case['expected']==c['expected']==full[c['case_index']]
 assert rows['selected-vm/'+c['label']+'.bin']['sha256']==c['input_sha256']==case['sha256']
 assert data('selected-vm/'+c['label']+'.stdout')==(case['expected']+'\n').encode() and data('selected-vm/'+c['label']+'.stderr')==b''
 assert rows[mapped(c['argv'][0])]['sha256']==micro['tool_before']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git(name):
 proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in q['source_before'].items():assert sha(git(str(Path(name).relative_to(m['roots']['comparison']))))==h
for name,h in q['production_source_before'].items():assert sha(git(name.split('/source/',1)[1]))==h
for name,h in parse['source_before'].items():assert sha(git(name))==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; all281 actual C commands, original1001/new3317 complete cache-path reports, 12 selected VM controls, exact source/tool/object guards, ordinary imported shadows and fresh-body/caller-binding protocol refusals. Full3317-case VM/source/sanitizer/paired, fresh callee/whole-walk/CFG/emission/release acceptance remain pending.'%len(rows))
