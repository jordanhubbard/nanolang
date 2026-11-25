"""I verify actual C effect-transfer expectations without claiming a NanoLang route."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'92d5bf4eb1a30e099d149074979983ae235c0de2fdf68d235ea8f44dae14d66c')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('comparison/result.json')
assert m['source_commit']==q['source_pin']=='c3e6017fb5235bac66e25aa1e8afb82b6f26c138'
assert m['component_only'] and m['c_oracle_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release'])
assert q['terminal'] and q['accepted_c_oracle'] and q['all_guards_match'] and len(q['commands'])==878
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
for group,count,h in [('original',1001,'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'),('effects',8128,'0521f0169aa763ebe53c45311cb89ff343c42589d22deea3b3cac3f5a257cba9')]:
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
assert parse['source_before']==parse['source_after'] and parse['compiler_sha256']==parse['compiler_after']
assert len(parse['commands'])==4 and all(c['exit']==0 for c in parse['commands'])
assert 'no NanoLang shadow execution' in parse['scope']
for name,h in parse['source_before'].items():
 assert rows['source-review/'+name]['sha256']==h
 if name.endswith(('.nano','.c')):assert rows['comparison/'+name]['sha256']==h
assert origin['images_before']==origin['images_after']
assert origin['images_after']['bin/nanoc_c']==origin['products']['bin/nanoc_c']['sha256']==parse['compiler_sha256']
assert b"Program must define a 'main' function" in data('source-review/nanolang-effect-draft.stderr')
# My state diagnostic includes the unchanged implementation instead of its object.
assert not any(s.endswith('/nanoisa/affine_state.o') or s.endswith('/nanoisa/affine_bytecode.o') for s in build)
assert sum(s.endswith('/tests/nvm2c_affine_owners_state.c') for s in build)==1
assert b'#include "../src/nanoisa/affine_state.c"' in data('comparison/tests/nvm2c_affine_owners_state.c')

repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git(name):
 proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in q['source_before'].items():assert sha(git(str(Path(name).relative_to(m['roots']['comparison']))))==h
for name,h in q['production_source_before'].items():assert sha(git(name.split('/source/',1)[1]))==h
for name,h in parse['source_before'].items():
 proc.stdin.write(('c3e6017fb5235bac66e25aa1e8afb82b6f26c138:'+name+'\n').encode());proc.stdin.flush()
 parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n'
 assert sha(raw)==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; all878 actual C commands, original1001/new8128 complete effect/key/escape reports and independent expectations, exact selected-source/tool/object guards, typed-driver/shadow-presence checks and retained library-only missing-main diagnostic. No NanoLang execution/five-route/sanitizer/paired/CFG/cutover/release acceptance.'%len(rows))
