"""I verify my failed global/array terminal, source controls and original limits."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'c7de9987d49abaac2dbe17732fa15d036919fc85262004633602e4ae1db6369a')
sha=lambda b:hashlib.sha256(b).hexdigest()
q=js('comparison/cohort-result.json');s=js('comparison/result.json')
assert not m['accepted_component'] and not m['full_five_routes'] and not m['whole_release'] and not m['interrupted']
assert m['source_commit']==q['source_pin']=='5d809aa3d14d11b25adc2ee2ae391893ae5cf0f2'
assert q['terminal'] and not q.get('accepted_component',False) and q['all_guards_match']
assert q['tests']==dict(tests=2,failures=2,errors=0,skips=0,passed=False)
assert q['original_methods']==0 and q['new_methods']==2 and q['ordinary_seconds']==10
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
assert len(q['providers'])==2
for n,p in q['providers'].items():
 assert p['read_only'] and p['before_translation'] and p['module_sha256']==p['module_after']==rows[mapped(n)]['sha256']
 assert p['libraries_before']==p['libraries_after']
 for path,h in p['libraries_before'].items():assert rows[mapped(path)]['sha256']==h

# I independently replay every command update and the complete terminal commit.
assert q['observer_before']==q['observer_after']
for name,h in q['observer_before'].items():assert rows[mapped(name)]['sha256']==h
# I bind all four actual observer scripts to the published current-source preparation.
prepared=js('comparison/admission-result.json')
assert prepared['terminal']
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

commands=q['commands'];assert len(commands)==8
assert [c['exit'] for c in commands]==[0,0,0,0,1,0,0,1]
assert all('interrupted_or_timeout' not in c for c in commands)
assert all(c['timeout']==600 for c in commands if c['build'])
for number,function in [(5,14),(8,19)]:
 c=commands[number-1];assert c['argv'][0].endswith('/bin/nvm2c')
 text=data('comparison/command-%04d.stderr'%number).decode()
 assert 'narr_t versus nmap_value at result field 2 of function %d'%function in text and 'offset 545' in text
 assert rows[mapped(c['argv'][1])]['bytes']>0
assert not any(n.endswith('.reports') for n in rows if n.startswith('comparison/tmp/'))
control=js('localization/comparison.json');assert control['terminal'] and control['all_guards_match']
assert control['diagnostic_only'] and not control['whole_release']
assert control['tools_before']==control['tools_after']
for n,h in control['tools_before'].items():assert rows['comparison/source/bin/'+n]['sha256']==h
assert len(control['commands'])==12
for i,c in enumerate(control['commands']):
 assert c['timeout']==600 and c['exit']==(i%2)
 if i%2:
  text=data('localization/'+c['label']+'.stderr').decode()
  assert 'narr_t versus nmap_value at result field 2' in text and 'offset 545' in text
for pin,files in control['sources'].items():
 for n,h in files.items():assert rows['localization/'+pin+'/'+n]['sha256']==h
# I bind copied inputs to actual Git objects independently of the runner's guards.
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in q['source_before'].items():
 proc.stdin.write((q['source_pin']+':'+n+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
 raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,n
for pin,files in control['sources'].items():
 for n,h in files.items():
  proc.stdin.write((pin+':'+n+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
  raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,n
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; exact failed two-method/eight-command terminal, six ordinary builds/six retained refusals, journal replay and immutable source/tool/host/object/provider guards. No report matrix or release acceptance.'%len(rows))
