"""I verify actual C local-transfer expectations without claiming a NanoLang route."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'b2485614d5281a25f8a46c6f7ca8a52243b9dbbdfbd22f5be0568d98d3b575b9')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('comparison/result.json')
assert m['source_commit']==q['source_pin']=='b43042ec0121c8c43f57c1205151ad1aaba1847b'
assert m['component_only'] and m['c_oracle_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release'])
assert q['terminal'] and q['accepted_c_oracle'] and q['all_guards_match'] and len(q['commands'])==761
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
origin=js('origin/bootstrap-result.json');assert origin['terminal'] and origin['original_make_pass'] and origin['source_pin']=='95a4062efebe0e97d8cde271117eac1deab21595'
for i,c in enumerate(q['commands'],1):assert c['exit']==0 and c['timeout'] in [60,600] and data('comparison/command-%04d.stderr'%i)==b''
for group,count,h in [('original',1001,'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'),('locals',10409,'92ba48b31372b47c07035c7e676e81e8a0ef633d08f1b57d071107ab1ef3a78d')]:
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
   valid,message,state=text.split('|',2);local,stack,global_,context=state.split('/')
   assert valid==str(int(expected['valid'])) and message==expected['message'] and stack==expected['stack'] and global_=='9,' and context=='1,23,5,7,',case['name']
   slots=local.strip(',').split(',')
   for index,live in expected['live'].items():assert slots[int(index)*4]==str(int(live)),case['name']
 if group=='locals':assert sha(b''.join(data(mapped(c['path'])) for c in corpus))=='3a60b1144d7605c4a1ad62f82435c73b760da870d0df6fdf7cca67904afaa64e'
red=js('harness-draft/result.json');green=js('harness-draft/review-result.json')
assert red['syntax_exit']==1 and green['corrected_syntax_exit']==0 and not green['executable_qualification']
assert b'redefinition of' in data('harness-draft/syntax.stderr') and b'main' in data('harness-draft/syntax.stderr')
assert data('harness-draft/corrected-syntax.stderr')==b''
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git(name):
 proc.stdin.write((q['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in q['source_before'].items():assert sha(git(str(Path(name).relative_to(m['roots']['comparison']))))==h
for name,h in q['production_source_before'].items():assert sha(git(name.split('/source/',1)[1]))==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; all761 actual C commands, original1001/new10409 complete reports and expectations, exact source/tool/object guards and retained strict-C harness failure. No NanoLang execution/five-route/sanitizer/paired/CFG/cutover/release acceptance.'%len(rows))
