"""I verify complete affine component reports, exact providers and original limits."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'994bb06c1eebb5c9034115ae291fd5a69ccabc2bd697dce531403af1c42d14fb')
sha=lambda b:hashlib.sha256(b).hexdigest()
q=js('comparison/cohort-result.json');s=js('comparison/result.json')
assert not m['accepted_component'] and not m['full_five_routes'] and m['completed_methods']==4 and not m['whole_release'] and not m['interrupted']
assert m['source_commit']==q['source_pin']=='b43042ec0121c8c43f57c1205151ad1aaba1847b'
assert q['terminal'] and not q.get('accepted_component') and q['all_guards_match']
assert q['tests']==dict(tests=5,failures=1,errors=0,skips=0,passed=False)
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
assert len(q['providers'])==8
for n,p in q['providers'].items():
 assert p['read_only'] and p['before_translation'] and p['module_sha256']==p['module_after']==rows[mapped(n)]['sha256']
 assert p['libraries_before']==p['libraries_after']
 for path,h in p['libraries_before'].items():assert rows[mapped(path)]['sha256']==h
commands=q['commands'];assert len(commands)==15237
assert all(c['exit']==0 and 'interrupted_or_timeout' not in c for c in commands[:-1])
failed=commands[-1];assert failed['exit']==1 and failed['build'] and failed['timeout']==600 and 'interrupted_or_timeout' not in failed
assert failed['argv'][0].endswith('/bin/nano_virt') and failed['argv'][1].endswith('/tests/nvm2c_affine_locals_driver.nano')
assert b'BOOL_NOT requires a boolean' in data('comparison/command-15237.stderr')
assert b'$shadow_203_transfer' in data('comparison/command-15237.stderr')
assert data('comparison/command-15237.stdout')==b''
builds=[c for c in commands if c['build']];assert len(builds)==30 and all(c['timeout']==600 for c in builds)
cc=[c for c in builds if '-fsanitize=address,undefined' in c['argv']];assert len(cc)==8
for c in cc:assert '-fno-sanitize-recover=all' in c['argv'] and '-std=c11' in c['argv']
source=[c for c in builds if c['argv'][0]==s['producer']];assert len(source)==4 and all('--emit-nvm' in c['argv'] for c in source)
for c in commands:
 assert c['timeout'] in [60,600]
 if c['build'] or c['number']<=3:continue
 paths=[a for a in c['argv'] if a.endswith('.bin')];assert paths
 new=Path(paths[0]).name.startswith(('scalar-','locals-'))
 if new:assert c['timeout']==60 and len(paths)<=32 and (len(paths)==1 or sum(rows[mapped(n)]['bytes'] for n in paths)<=8192)
 else:assert len(paths)<=8
labels={'actual-c','c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native'}
counts=[];reports=0;incomplete=0
for n in rows:
 if not n.startswith('comparison/tmp/') or not n.endswith('/manifest.json'):continue
 spec=js(n);folder=n.rsplit('/',1)[0]
 if not spec['routes']:
  assert 'nano-affine-locals-' in folder and len(spec['cases'])==11410
  for c in spec['cases']:assert rows[mapped(c['path'])]['sha256']==c['sha256']
  incomplete+=1;continue
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
assert sorted(counts)==[336,1001,1001,3598,25643] and incomplete==1
assert reports==189474
log=data('comparison/permanent-test.stderr');assert log.count(b' ... ok\n')==4 and b'FAILED (failures=1)' in log
# I bind copied inputs to actual Git objects independently of the runner's guards.
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in q['source_before'].items():
 proc.stdin.write((q['source_pin']+':'+n+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
 raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,n
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; exact failed five-method terminal,15236 successful commands plus retained local-shadow refusal, four completed methods/%d complete reports, all guards and original budgets. No local/five-method/full-foundation/CFG/private/paired/pure emission/cutover/release acceptance.'%(len(rows),reports))
