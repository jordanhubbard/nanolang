"""I verify my directed original-frame fixture controls."""
from pathlib import Path
import hashlib,json,sys,subprocess
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,records,data,js=read_archive(base,"1cce1d7d6614bef2cfb49980e9bbc474cf22d68c707acf4f7be23808250cb94f")
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['source_commit']=='c886e6dc73c2680d17a6db76b18e4ec00050e592'
assert m['working_overlay'] and m['accepted_fixture_report_controls']
assert not any(m[k] for k in ['accepted_component','full_five_routes','whole_release','accepted_public_c_vm'])
q=js('review/controls-result.json');r=js('review/result.json')
assert q['terminal'] and q['accepted_old_report_controls'] and q['all_guards_match']
assert r['terminal'] and r['accepted_ordinary_shadows'] and r['all_guards_match']
assert q['ordinary_seconds']==r['ordinary_seconds']==10
assert q['review_sha256']==sha(data('review/result.json'))
assert len(q['commands'])==1143 and len(r['commands'])==6 and len(q['reports'])==9
root=Path(m['roots']['review'])
def mapped(n):
 p=Path(n)
 if p.is_relative_to(root):return 'review/'+str(p.relative_to(root))
 return 'external/'+str(p).lstrip('/')
for state,groups in [(r,['source','tools']),(q,['source','tools','hosts','input'])]:
 for group in groups:
  assert state[group+'_before']==state[group+'_after']
  for n,h in state[group+'_before'].items():
   target='review/source/'+n if group=='source' else ('inputs/'+n.lstrip('/') if group=='input' else mapped(n))
   assert records[target]['sha256']==h,target
 for c in state['commands']:
  assert c['exit']==0 and not c.get('timed_out')
  assert c['timeout'] in [60,600]
assert q['source_before']==r['source_before']
assert len(q['input_before'])==3003
expected='f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
seen=set();total=0
for leaf in ['aggregates','effects','value_calls']:
 old=js('old-'+leaf+'/manifest.json');cases=[c for c in old['cases'] if c['group']=='original']
 assert len(cases)==1001
 baseline=data('old-'+leaf+'/original-actual-c.reports');assert sha(baseline)==expected
 bad=data('old-'+leaf+'/original-c-seed-native.reports')
 a=baseline.splitlines();b=bad.splitlines();assert len(a)==len(b)==1001
 diffs=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
 assert len(diffs)==665 and diffs==list(range(336,1001))
 assert all(b[i].startswith(a[i]+b'|') for i in diffs)
 assert data('review/'+leaf+'-old-positive.stdout').rstrip(b'\n')==b[336]
 for route in ['native','vm','genuine-vm']:
  raw=data('review/'+leaf+'-'+route+'.reports');assert raw==baseline
  row=next(x for x in q['reports'] if x['leaf']==leaf and x['route']==route)
  assert row['cases']==1001 and row['sha256']==expected
  commands=[c for c in q['commands'] if c['label'].startswith(leaf+'-'+route+'-') and c['label'].rsplit('-',1)[-1].isdigit()]
  assert len(commands)==126
  assert b''.join(data('review/'+c['label']+'.stdout') for c in commands)==raw
  paths=[]
  for c in commands:
   prefix=3 if route!='native' else 1
   inputs=c['argv'][prefix:];assert 1<=len(inputs)<=8 and c['timeout']==600
   paths.extend(inputs);assert c['label'] not in seen;seen.add(c['label'])
  assert paths==[c['path'] for c in cases]
  total+=len(cases)
assert total==9009 and len(seen)==1134
for leaf in ['aggregates','effects','value_calls']:
 name='tests/nvm2c_affine_'+leaf+'_driver.nano'
 pinned=subprocess.check_output(['git','show',m['source_commit']+':'+name],cwd=base)
 assert pinned==data('review/source/'+name)
 old=data('old-'+leaf+'/driver.nano')
 # I preserve all functions outside the corrected report and its shadow.
 start=old.index(b'fn describe_frames(');end=old.index(b'\nshadow describe_frames',start)
 new=pinned;ns=new.index(b'fn describe_frames(');ne=new.index(b'\nshadow describe_frames',ns)
 assert old[:start]==new[:ns]
 # The original reporter loses only the extra authority suffix.
 before=old[start:end];after=new[ns:ne]
 assert before!=after
 oldtail=old[old.index(b'\nfn ',end):]
 newtail=new[new.index(b'\nfn ',ne):]
 assert oldtail==newtail
parent=js('failed-six/cohort-result.json')
assert parent['terminal'] and parent['all_guards_match'] and not parent.get('accepted_component',False)
assert parent['tests']['tests']==6 and parent['tests']['failures']==4
assert len(parent['commands'])==11093
failed=[c for c in parent['commands'] if c['exit']!=0];assert len(failed)==1 and failed[0]['number']==11093
assert b'I require the matching scalar shape at this array write' in data('failed-six/command-11093.stderr')
print(json.dumps(dict(terminal=True,accepted_fixture_report_controls=True,reports=total,commands=1149,full_five_routes=False,whole_release=False)))
