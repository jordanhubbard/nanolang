"""I verify actual C value-call expectations without claiming a NanoLang route."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'a8eff0780df320b4c10204f6d6492930fddc5a9992989be613dec22232ca87a2')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('comparison/result.json')
assert m['source_commit']==q['source_pin']=='121d1b58b06d788a94c4654faf79bdd3414df26c'
assert m['component_only'] and m['c_oracle_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release'])
assert q['terminal'] and q['accepted_c_oracle'] and q['all_guards_match'] and len(q['commands'])==1255
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
for group,count,h in [('original',1001,'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'),('calls',15644,'9e7178f7a0944609b3486cd53cb528232aa8cf5125282cab14ebdf7df45c1fec')]:
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
   valid,message,frame_,authority_,allowed,keys,derived=text.split('|',6);local,stack,globals_,context=frame_.split('/')
   assert valid==str(int(expected['valid'])) and message==expected['message'] and stack==expected['stack'],case['name']
   assert globals_==expected['globals'] and context==expected['context'] and authority_==expected['authority'] and keys==expected['keys'],case['name']
   assert allowed==expected['allowed'] and derived==str(int(expected['derived'])),case['name']
parse=js('source-review/result.json');assert parse['terminal'] and parse['all_guards_match']
assert parse['source_before']==parse['source_after'] and parse['tools_before']==parse['tools_after']
assert len(parse['commands'])==5 and all(c['exit']==0 for c in parse['commands'])
assert parse['commands'][-1]['timeout']==600 and parse['commands'][-1]['ordinary_shadow_seconds']==10
assert 'ordinary C-producer NanoVirt imported shadows only' in parse['scope']
for name,h in parse['source_before'].items():
 assert rows['source-review/'+name]['sha256']==h
 if name.endswith(('.nano','.c')):assert rows['comparison/'+name]['sha256']==h
for name,h in parse['tools_before'].items():assert rows[mapped(name)]['sha256']==h
assert origin['images_before']==origin['images_after']
# I retain every failed control, not only the corrected report.
failed=js('failed-expectation/result.json');assert failed['terminal'] and failed['all_guards_match'] and not failed.get('accepted_c_oracle')
assert len(failed['commands'])==1255 and all(c['exit']==0 for c in failed['commands'])
assert failed['groups'][0]['failures']==[] and len(failed['groups'][1]['failures'])==3
assert {f['name'] for f in failed['groups'][1]['failures']}=={'graph-target-False-1-2','graph-target-False-1-8','graph-target-False-7-8'}
old=js('failed-expectation/calls-manifest.json');new=js('comparison/calls-manifest.json');assert len(old)==len(new)==15644
changed=0
for before,after in zip(old,new):
 assert before['name']==after['name'] and before['expected']==after['expected']
 left=data(mapped(before['path']));right=data(mapped(after['path']))
 if left!=right:
  assert before['name'].startswith('graph-target-False-')
  assert left[:12]==right[:12] and left[16:]==right[16:] and left[12:16]==b'\x01\x00\x00\x00' and right[12:16]==bytes(4)
  changed+=1
assert changed==10
before_reports=data('failed-expectation/calls.reports').decode().splitlines();after_reports=data('comparison/calls.reports').decode().splitlines()
assert len(before_reports)==len(after_reports)==15644
assert {c['name'] for c,left,right in zip(new,before_reports,after_reports) if left!=right}=={f['name'] for f in failed['groups'][1]['failures']}
for before,after in zip(old,before_reports):assert before['sha256']==rows[mapped(before['path'])]['sha256']
for name in ['tests/test_nanolang_translator_affine_value_calls.py']:
 a=data('source-review/'+name);b=data('comparison/'+name)
 assert a.replace(b"add(f'graph-target-{graph}-{target}-{functions}',target=target,functions=functions,",b"add(f'graph-target-{graph}-{target}-{functions}',graph=graph,target=target,functions=functions,")==b

controls=[c for c in new if c['name'].startswith('uncached-')];assert len(controls)==12 and all(c['expected']['derived'] for c in controls)
reference=data('comparison/tests/nvm2c_affine_value_calls_reference.c')
assert b'calls->results[target].ok==(ok!=0)' in reference and b'calls->escape_count[target]==callee_count' in reference
assert b'memcmp(calls->global_allowed[target],callee_allowed,globals)' in reference and b'memcmp(calls->global_returns[target],callee_returns,globals)' in reference
strict=js('failed-strict/result.json');assert b'misleading-indentation' in data('failed-strict/strict.stderr')
shadow=js('failed-shadow/result.json');assert shadow['terminal'] and shadow['all_guards_match']
assert shadow['commands'][-1]['exit']==1 and all(c['exit']==0 for c in shadow['commands'][:-1])
assert b'$shadow_223_prepare' in data('failed-shadow/ordinary-shadow-module.stderr') and b':53:26' in data('failed-shadow/ordinary-shadow-module.stderr')
assert b'assert ready.handled assert ready.valid' in data('failed-shadow/src_nano/nvm2c/affine_value_calls.nano')
for opt in ['-O3','-O0']:
 assert data('graph-diagnosis/'+opt+'run.stderr').count(b'before graph=1 calls=1')==3
 assert data('graph-diagnosis/'+opt+'run.stdout')==b''.join((f['actual']+'\n').encode() for f in failed['groups'][1]['failures'])
# My state diagnostic includes the unchanged implementation instead of its object.
assert not any(s.endswith('/nanoisa/affine_state.o') or s.endswith('/nanoisa/affine_bytecode.o') for s in build)
assert sum(s.endswith('/tests/nvm2c_affine_owners_state.c') for s in build)==1
assert b'#include "../src/nanoisa/affine_state.c"' in data('comparison/tests/nvm2c_affine_owners_state.c')

repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git(name):
 proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in q['source_before'].items():assert sha(git(str(Path(name).relative_to(m['roots']['comparison']))))==h
for name,h in q['production_source_before'].items():assert sha(git(name.split('/source/',1)[1]))==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; all1255 actual C commands, original1001/new15644 complete value-call reports, 12 uncached C-derived summary controls, exact source/tool/object guards, ordinary imported shadows and retained strict/shadow/graph-fixture failures. Pure NanoLang callee analysis, full-route/sanitizer/paired/CFG/emission/release acceptance remain pending.'%len(rows))
