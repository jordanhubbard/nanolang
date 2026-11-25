"""I verify actual C local-transfer expectations without claiming a NanoLang route."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'5996d4afd5727864c9c11b4c0dc7b19a6f32d12bb90361da40cc187f70768e5f')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('comparison/result.json')
assert m['source_commit']==q['source_pin']=='be8047814d3ee10b9274f91025924ba7e431b522'
assert m['component_only'] and m['c_oracle_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release'])
assert q['terminal'] and q['accepted_c_oracle'] and q['all_guards_match'] and len(q['commands'])==316
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
for group,count,h in [('original',1001,'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'),('globals',3688,'e5bc1a381ceddd87cbe704d0e38d4ff341eb886aff156ad0c0d7b4b9597b931c')]:
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
   valid,message,state,allowed=text.split('|',3);local,stack,global_,context=state.split('/')
   assert valid==str(int(expected['valid'])) and message==expected['message'] and stack==expected['stack'] and context=='1,23,5,7,',case['name']
   assert global_==''.join(f'{v},' for v in expected['relations']) and allowed==''.join(f'{v},' for v in expected['allowed']),case['name']
 if group=='globals':assert sha(b''.join(data(mapped(c['path'])) for c in corpus))=='d08c42b3f9008c259727c49a81f76f31b0093edce0abb529ee7aa7603edc7d6c'
parse=js('source-review/result.json');corrected=js('source-review/corrected-result.json')
assert parse['commands'][0]['exit']==0 and parse['commands'][1]['exit']==1 and corrected['exit']==0 and not corrected['shadow_execution']
assert b'Unexpected token in expression: EFFECT' in data('source-review/typed-ast.stderr')
include=js('include-failure/result.json');assert include['terminal'] and include['commands'][0]['exit']==1 and not include['groups']
assert b'affine_bytecode.c: No such file or directory' in data('include-failure/command-0001.stderr')
red=js('expectation-failure/result.json');assert red['terminal'] and red['all_guards_match'] and len(red['groups'])==2
assert len(red['groups'][1]['failures'])==1 and red['groups'][1]['failures'][0]['name']=='valid-malformed-base'
assert red['groups'][1]['report_sha256']==q['groups'][1]['report_sha256']
old=js('expectation-failure/globals-manifest.json');new=js('comparison/globals-manifest.json')
assert [c['sha256'] for c in old]==[c['sha256'] for c in new]
assert sum(a['expected']!=b['expected'] for a,b in zip(old,new))==1
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git(name):
 proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in q['source_before'].items():assert sha(git(str(Path(name).relative_to(m['roots']['comparison']))))==h
for name,h in q['production_source_before'].items():assert sha(git(name.split('/source/',1)[1]))==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; all316 actual C commands, original1001/new3688 complete reports and expectations, exact source/tool/object guards and retained reserved-name, include-view and oracle-expectation failures. No NanoLang execution/five-route/sanitizer/paired/CFG/cutover/release acceptance.'%len(rows))
