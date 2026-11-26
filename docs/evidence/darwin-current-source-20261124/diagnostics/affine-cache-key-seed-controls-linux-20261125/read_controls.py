"""I verify my typed key-row fixture controls without admitting the full component."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,records,data,js=read_archive(base,'3fd80a1344ff25b6ccfa6cfe689e8bafce613b773cb879c42b4583bcf998f73e')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['source_commit']=='901caef8e6f995080ac7d3c48c4cf23c75c99315' and m['accepted_fixture_four_route_controls']
assert not any(m[n] for n in ['accepted_component','whole_release','full_five_routes'])
q=js('review/controls/cohort-result.json');r=js('review/result.json');cases=js('cases/manifest.json')
assert q['terminal'] and q['accepted_four_route_report_controls'] and q['all_guards_match']
assert r['terminal'] and r['accepted_build_shadow_translation_controls'] and r['all_guards_match']
assert q['review_sha256']==sha(data('review/result.json')) and q['case_manifest_sha256']==sha(data('cases/manifest.json'))
assert q['ordinary_seconds']==r['ordinary_seconds']==10 and len(q['commands'])==1120 and len(r['commands'])==6
root=Path(m['roots']['review'])
def mapped(n):
 p=Path(n)
 return 'review/'+str(p.relative_to(root)) if p.is_relative_to(root) else 'external/'+str(p).lstrip('/')
for state,groups in [(r,['source','tools']),(q,['source','tools','hosts','input','modules'])]:
 for group in groups:
  assert state[group+'_before']==state[group+'_after']
  for n,h in state[group+'_before'].items():
   name='review/source/'+n if group=='source' else 'inputs/'+n.lstrip('/') if group=='input' else mapped(n)
   assert records[name]['sha256']==h,name
 for c in state['commands']:assert c['exit']==0 and not c.get('timed_out') and not c.get('timeout_expired')
assert q['source_before']==r['source_before'] and len(q['input_before'])==4318 and len(q['modules_before'])==2
# I reconstruct every report in exact packet order at the original limits.
seen=set();total=0
for group,count in [('original',1001),('cache',3317)]:
 selected=[c for c in cases['cases'] if c['group']==group];assert len(selected)==count
 baseline=data('review/controls/'+group+'-actual-c.reports');lines=baseline.decode().splitlines();assert len(lines)==count
 assert sha(baseline)==('f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0' if group=='original' else '2a6203cea85a5303b9907da8f082a9201717252edaa0516c3b435f7bf217d53c')
 for case,line in zip(selected,lines):
  assert q['input_before'][case['path']]==case['sha256']
  if group=='cache' or case['name'].startswith(('global-algebra-','bounded-fifo-')):assert line==case['expected']
  else:assert line.startswith(case['expected'])
 for route in ['actual-c','c-seed-native','c-producer-vm','genuine-producer-vm']:
  raw=data('review/controls/'+group+'-'+route+'.reports');assert raw==baseline
  row=next(x for x in q['reports'] if x['group']==group and x['route']==route);assert row['cases']==count and row['sha256']==sha(raw)
  paths={c['path'] for c in selected};prefixlen=3 if route.endswith('-vm') else 1
  commands=[]
  for c in q['commands']:
   args=c['argv'];image=Path(args[0]).name
   matches=(route=='actual-c' and image=='analysis_cache-reference') or (route=='c-seed-native' and image=='decoder-native') or (route=='c-producer-vm' and len(args)>1 and Path(args[1]).name=='decoder.nvm') or (route=='genuine-producer-vm' and len(args)>1 and Path(args[1]).name=='source-decoder.nvm')
   if matches and args[prefixlen] in paths:commands.append(c)
  assert [n for c in commands for n in c['argv'][prefixlen:]]==[c['path'] for c in selected]
  assert b''.join(data('review/controls/command-%04d.stdout'%c['number']) for c in commands)==raw
  for c in commands:
   inputs=c['argv'][prefixlen:];assert c['timeout']==(600 if group=='original' else 60)
   assert len(inputs)<=(8 if group=='original' else 32)
   if group=='cache':assert len(inputs)==1 or sum(records['inputs/'+n.lstrip('/')]['bytes'] for n in inputs)<=8192
   assert c['number'] not in seen;seen.add(c['number'])
  total+=count
assert len(q['reports'])==8 and total==17272 and len(seen)==1120
state=None;history=[];committed=False
for i,line in enumerate(data('review/controls/cohort-journal.jsonl').splitlines(),1):
 e=json.loads(line);assert e['sequence']==i and not committed
 if e['kind']=='metadata':state=e['value'];assert state['commands']==[]
 elif e['kind']=='command':
  n=e['number'];assert n==e['value']['number'] and n in [len(history),len(history)+1] and n>0
  if n==len(history)+1:history.append(e['value'])
  else:history[-1]=e['value']
 elif e['kind']=='commit':
  state['commands']=history;raw=(json.dumps(state,indent=2)+'\n').encode();assert state['terminal'] and e['commands']==len(history)
  assert raw==data('review/controls/cohort-result.json') and len(raw)==e['bytes'] and sha(raw)==e['sha256'];committed=True
 else:raise AssertionError('I require a known journal event')
assert committed and state==q
name='tests/nvm2c_affine_analysis_cache_driver.nano';old=data('review/original-driver.nano');new=data('review/source/'+name)
assert new==subprocess.check_output(['git','show',m['source_commit']+':'+name],cwd=base)
assert old==subprocess.check_output(['git','show','67d99e5121521428740d385d10071b9679e66013:'+name],cwd=base)
a=old.index(b'fn describe_cache(');b=new.index(b'fn describe_cache(');assert old[:a]==new[:b]
a=old.index(b'\nfn describe(bytes:');b=new.index(b'\nfn describe(bytes:');assert old[a:]==new[b:]
a=old.index(b'    let before: string = (cache_snapshot input)',old.index(b'fn describe_cache('));b=new.index(b'    let before: string = (cache_snapshot input)',new.index(b'fn describe_cache('))
assert old[a:old.index(b'\nshadow describe_cache',a)]==new[b:new.index(b'\nshadow describe_cache',b)]
assert b'let mut prepared_keys: array<array<int>> = []' in new and b'keys: prepared_keys' in new
print(json.dumps(dict(terminal=True,accepted_fixture_four_route_controls=True,reports=total,commands=1126,full_five_routes=False,whole_release=False)))
