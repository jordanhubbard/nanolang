"""I verify complete affine component reports, exact providers and original limits."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'b4107c3de1b250800f1b519f5353b82d08930cc0edba766d08eecaf128e3df6d')
sha=lambda b:hashlib.sha256(b).hexdigest()
q=js('comparison/cohort-result.json');s=js('comparison/result.json')
assert m['accepted_component'] and m['full_five_routes'] and not m['whole_release'] and not m['interrupted']
assert m['source_commit']==q['source_pin']=='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8'
assert q['terminal'] and q['accepted_component'] and q['all_guards_match']
assert q['tests']==dict(tests=5,failures=0,errors=0,skips=0,passed=True)
assert q['original_methods']==3 and q['new_methods']==2 and q['ordinary_seconds']==10
root=Path(m['roots']['comparison'])
def mapped(name):
 p=Path(name)
 return 'comparison/'+str(p.relative_to(root)) if p.is_relative_to(root) else 'external/'+str(p).lstrip('/')
for group in ['source','images','tools','hosts']:
 assert q[group+'_before']==q[group+'_after']
 for n,h in q[group+'_before'].items():assert rows['comparison/source/'+n if group in ['source','images'] else mapped(n)]['sha256']==h
assert q['object_inputs']==q['objects_after']
for n,h in q['object_inputs'].items():assert rows[mapped(n)]['sha256']==h
assert q['producer_sha256']==q['producer_after']==s['producer_sha256']=='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
assert rows[mapped(s['producer'])]['sha256']==s['producer_sha256']
o=js('origin/bootstrap-result.json');assert o['terminal'] and o['original_make_pass'] and o['source_pin']==q['producer_source_pin']==s['bootstrap_source_pin']
assert sha(data('origin/bootstrap-result.json'))==s['bootstrap_result_sha256']
assert len(q['providers'])==10
for n,p in q['providers'].items():
 assert p['read_only'] and p['before_translation'] and p['module_sha256']==p['module_after']==rows[mapped(n)]['sha256']
 assert p['libraries_before']==p['libraries_after']
 for path,h in p['libraries_before'].items():assert rows[mapped(path)]['sha256']==h

# I independently replay every command update and the complete terminal commit.
assert q['observer_before']==q['observer_after']
for name,h in q['observer_before'].items():assert rows[mapped(name)]['sha256']==h
preparation='docs/evidence/darwin-current-source-20261124/diagnostics/affine-local-shadow-localization-linux-20261125/preparation/'
prepared=json.loads(subprocess.check_output(['git','show','16687bbf02d947b3046eb064e6e159d060c93c42:'+preparation+'manifest.json'],cwd=base))
for name,h in q['observer_before'].items():assert prepared['files'][Path(name).name]==h
assert s['preflight_evidence']['observer_helper_sha256']==q['observer_before'][str(root/'cohort_journal.py')]
state=None;history=[];committed=False;commits=0
for number,line in enumerate(data('comparison/cohort-journal.jsonl').splitlines(),1):
 event=json.loads(line);assert event['sequence']==number and not committed
 if event['kind']=='metadata':state=event['value'];assert state['commands']==[]
 elif event['kind']=='command':
  index=event['number'];row=event['value'];assert index==row['number']
  if index==len(history)+1:history.append(row)
  else:assert index==len(history);history[-1]=row
 else:
  assert event['kind']=='commit' and state is not None and state['terminal'] and event['commands']==len(history)
  state['commands']=history;raw=(json.dumps(state,indent=2)+'\n').encode()
  assert sha(raw)==event['sha256'] and len(raw)==event['bytes'] and raw==data('comparison/cohort-result.json')
  committed=True;commits+=1
assert committed and commits==1 and state==q

commands=q['commands'];assert all(c['exit']==0 and 'interrupted_or_timeout' not in c for c in commands)
builds=[c for c in commands if c['build']];assert len(builds)==35 and all(c['timeout']==600 for c in builds)
cc=[c for c in builds if '-fsanitize=address,undefined' in c['argv']];assert len(cc)==10
for c in cc:assert '-fno-sanitize-recover=all' in c['argv'] and '-std=c11' in c['argv']
source=[c for c in builds if c['argv'][0]==s['producer']];assert len(source)==5 and all('--emit-nvm' in c['argv'] for c in source)
for c in commands:
 assert c['timeout'] in [60,600]
 if c['build'] or c['number']<=3:continue
 paths=[a for a in c['argv'] if a.endswith('.bin')];assert paths
 new=Path(paths[0]).name.startswith(('scalar-','locals-'))
 if new:assert c['timeout']==60 and len(paths)<=32 and (len(paths)==1 or sum(rows[mapped(n)]['bytes'] for n in paths)<=8192)
 else:assert len(paths)<=8
labels={'actual-c','c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native'}
counts=[];reports=0
for n in rows:
 if not n.startswith('comparison/tmp/') or not n.endswith('/manifest.json'):continue
 spec=js(n);folder=n.rsplit('/',1)[0]
 if 'corpus' in spec:groups={'':spec['corpus']}
 else:
  cases=spec.get('groups',spec.get('cases'));groups={g:[c for c in cases if c['group']==g] for g in {c['group'] for c in cases}}
 for group,cases in groups.items():
  selected=[r for r in spec['routes'] if r.get('group','')==group];assert {r['label'] for r in selected}==labels and len(selected)==6
  baseline=data(folder+'/'+(group+'-' if group else '')+'actual-c.reports');lines=baseline.decode().splitlines();assert len(lines)==len(cases)
  if group=='original':assert sha(baseline)=='f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
  for c,text in zip(cases,lines):
   assert rows[mapped(c['path'])]['sha256']==c['sha256'];e=c['expected']
   if isinstance(e,str):
    if c['name'].startswith(('global-algebra-','bounded-fifo-')) or (group and group!='original'):assert text==e
    else:assert text.startswith(e)
   else:
    flag,message,state=text.split('|',2);local,stack,globals_,context=state.split('/')
    assert flag==str(int(e['valid'])) and message==e['message'] and stack==e['stack'] and globals_=='9,' and context=='1,23,5,7,'
    for index,live in e.get('live',{}).items():assert local.strip(',').split(',')[int(index)*4]==str(int(live))
  for route in selected:
   raw=data(folder+'/'+(group+'-' if group else '')+route['label']+'.reports');assert raw==baseline and sha(raw)==route['sha256']
   assert route.get('reports',route.get('cases'))==len(cases);reports+=len(cases)
  counts.append(len(cases))
assert sorted(counts)==[336,1001,1001,1001,3598,10409,25643] and reports==257934
# I bind copied inputs to actual Git objects independently of the runner's guards.
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in q['source_before'].items():
 proc.stdin.write((q['source_pin']+':'+n+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
 raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,n
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; all %d commands, five methods, %d complete reports, exact source/tool/host/provider/origin/observer guards, complete journal replay and original budgets. No CFG/private/paired/pure C emission/cutover/release acceptance.'%(len(rows),len(commands),reports))
