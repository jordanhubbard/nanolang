"""I verify actual C call-context initialization expectations without claiming a NanoLang route."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'7694814b347fd0dbe945b444f6f08955c04e838a59fbf9937b2db38bc91367ae')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('comparison/result.json')
assert m['source_commit']==q['source_pin']=='d992d95bb3566e6d4169bb08778a9bef466fd450'
assert m['component_only'] and m['c_oracle_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release'])
assert q['terminal'] and q['accepted_c_oracle'] and q['all_guards_match'] and len(q['commands'])==165
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
for group,count,h in [('original',1001,'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'),('context',834,'b0676f47e95f91952daab6a64008e14ec2342e67f5b60592f201c5f6ba331989')]:
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
assert not any(s.endswith('/tests/nvm2c_affine_owners_state.c') for s in build)
failed=js('failed-expectation/result.json');assert failed['terminal'] and failed['all_guards_match'] and not failed.get('accepted_c_oracle')
assert len(failed['commands'])==165 and all(c['exit']==0 for c in failed['commands'])
assert failed['groups'][0]['failures']==[] and len(failed['groups'][1]['failures'])==174
old=js('failed-expectation/context-manifest.json');new=js('comparison/context-manifest.json');assert len(old)==len(new)==834
import struct
word=lambda raw,n:struct.unpack_from('<I',raw,n)[0]
changed=0
for before,after in zip(old,new):
 assert before['name']==after['name'] and before['expected']==after['expected']
 left=data(mapped(before['path']));right=data(mapped(after['path']))
 if left==right or before['expected']=='D':continue
 # I decode only fixture framing; I do not derive any graph or ownership expectation.
 assert left[:12]==right[:12] and word(left,12)==len(left)-16 and word(right,12)==len(right)-16
 assert left[16:20]==right[16:20] and left[24:44]==right[24:44]
 at=44
 for i in range(word(left,24)):
  size=word(left,at);at+=4+size
 for i in range(word(left,28)):at+=32+word(left,at+28)
 at+=word(left,16)
 own=left[at:at+word(left,20)];other=right[at:at+word(right,20)]
 assert word(own,0)==3 and own[-12:]==struct.pack('<III',4,0,0)
 assert other==struct.pack('<I',1)+own[4:-12]
 assert left[44:at]==right[44:at] and left[at+len(own):]==right[at+len(other):]
 changed+=1
assert changed>174
before_reports=data('failed-expectation/context.reports').decode().splitlines();after_reports=data('comparison/context.reports').decode().splitlines()
assert len(before_reports)==len(after_reports)==834
assert {c['name'] for c,a,b in zip(new,before_reports,after_reports) if a!=b}=={f['name'] for f in failed['groups'][1]['failures']}
# My sole expectation-generator change is the ownership version selection.
name='tests/test_nanolang_translator_affine_call_context.py'
assert data('failed-expectation/'+name).replace(b'owned=record(3,functions=declarations,extensions=extensions)',b'owned=record(3 if extensions else 1,functions=declarations,extensions=extensions)')==data('comparison/'+name)
initial=js('initial-source-review/result.json');assert initial['terminal'] and initial['all_guards_match'] and all(c['exit']==0 for c in initial['commands'])
# I add only assertions to the initializer's existing shadow, retaining its algorithm.
name='src_nano/nvm2c/affine_call_context.nano'
assert data('failed-expectation/'+name).split(b'shadow create {',1)[0]==data('comparison/'+name).split(b'shadow create {',1)[0]
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git(name):
 proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in q['source_before'].items():assert sha(git(str(Path(name).relative_to(m['roots']['comparison']))))==h
for name,h in q['production_source_before'].items():assert sha(git(name.split('/source/',1)[1]))==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; all165 actual C commands, original1001/new834 complete context reports, exact source/tool/object guards, ordinary imported shadows, and retained174 version-three empty-extension fixture refusals with unchanged expectations. Pure callee/full-route/sanitizer/paired/CFG/emission/release acceptance remain pending.'%len(rows))
