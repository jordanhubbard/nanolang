"""I verify the unrepaired projection defect, typed fixture and offline observer controls."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'6101cb0c7d2fa544e8a63a15d814d9e977dfd88a23ae6f27af34b20abdc2b319')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['source_commit']=='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8' and m['component_only']
assert not any(m[k] for k in ['compiler_fixed','full_five_routes','new_bootstrap','whole_release','observer_replaced'])
def mapped(name):
 p=Path(name)
 for label,root in sorted(m['roots'].items(),key=lambda pair:len(pair[1]),reverse=True):
  if p.is_relative_to(root):return label+'/'+str(p.relative_to(root))
 return 'external/'+str(p).lstrip('/')
q=js('localization/result.json');assert q['terminal'] and q['all_guards_match'] and q['commands'][0]['exit']==1
assert q['source_before']==q['source_after'] and q['tools_before']==q['tools_after']
for name,h in {**q['source_before'],**q['tools_before']}.items():
 key=mapped(name)
 if key in rows:assert rows[key]['sha256']==h
 else:
  relative=name.split('/source/',1)[1]
  assert sha(subprocess.check_output(['git','show',q['source_pin']+':'+relative],cwd=base))==h
trace=js('localization/trace-result.json');assert trace['exit']==1 and trace['timeout']==600 and trace['ordinary_shadow_seconds']==10 and not trace['product']
assert trace['stderr_bytes']==rows['localization/original-traced.stderr']['bytes']==449371530
raw=data('localization/original-traced.stderr');tail=raw[-12000:]
assert b'fn=38(no_value)' in tail and b'top={tag=struct' in tail and b'fn=428($shadow_203_transfer) off=115058 op=BOOL_NOT' in tail
assert b'BOOL_NOT requires a boolean' in tail
mini=js('localization/minimal/result.json');assert mini['terminal'] and [c['exit'] for c in mini['commands']]==[1,0]
assert b'BOOL_NOT requires a boolean' in data('localization/minimal/direct.stderr')
assert data('localization/minimal/bound.stderr')==b''
for name,h in mini['source'].items():assert rows['localization/minimal/'+name]['sha256']==h
assert data('localization/minimal/direct.nano').replace(b'    assert (not (query.query).valid)',b'    let result: query.Query = (query.query)\n    assert (not result.valid)')==data('localization/minimal/bound.nano')
routes=js('localization/minimal/route-result.json');assert routes['terminal'] and routes['all_guards_match']
assert routes['tools_before']==routes['tools_after'] and len(routes['commands'])==8 and all(c['exit']==0 for c in routes['commands'])
for name,h in routes['tools_before'].items():assert rows[mapped(name)]['sha256']==h
assert sorted(c['label'] for c in routes['commands'])==sorted(r+'-'+n+s for r in ['c-seed','source'] for n in ['direct','bound'] for s in ['', '-run'])
for c in routes['commands']:assert c['timeout']==(60 if c['label'].endswith('-run') else 600)
assert rows[mapped(mini['commands'][0]['argv'][0])]['sha256']==mini['tool_sha256']
two=js('two-bindings/result.json');typed=js('typed-fixture/result.json')
assert two['exit']==typed['exit']==0 and two['product'] and typed['all_guards_match']
assert typed['before']==typed['after'] and typed['timeout']==600 and typed['ordinary_shadow_seconds']==10
for name,h in typed['before'].items():assert rows[mapped(name)]['sha256']==h
assert rows['typed-fixture/typed.nvm']['bytes']>0
source=data('typed-fixture/src_nano/nvm2c/affine_locals.nano');original=subprocess.check_output(['git','show','b43042ec0121c8c43f57c1205151ad1aaba1847b:src_nano/nvm2c/affine_locals.nano'],cwd=base)
expected=original.replace(b'    assert (not (state.local_info loaded.frame.locals 1).valid)',b'    let prior_info: state.Query = (state.local_info loaded.frame.locals 1)\n    assert (not prior_info.valid)').replace(b'    assert stored.valid assert (state.local_info stored.frame.locals 1).valid',b'    let stored_info: state.Query = (state.local_info stored.frame.locals 1)\n    assert stored.valid assert stored_info.valid').replace(b'    assert (not (state.local_info floating.locals 1).valid)',b'    let floating_info: state.Query = (state.local_info floating.locals 1)\n    assert (not floating_info.valid)')
assert expected==source==subprocess.check_output(['git','show',m['source_commit']+':src_nano/nvm2c/affine_locals.nano'],cwd=base)
assert original.count(b'assert ')==source.count(b'assert ')
assert data('static-source/codegen.c')==subprocess.check_output(['git','show',m['source_commit']+':src/nanovirt/codegen.c'],cwd=base)
# I independently replay the complete retained snapshot, without importing its writer.
measurement=js('observer-corrected/measurement.json');controls=js('observer-corrected/controls.json');timing=js('observer-corrected/journal-measurement.json')
assert measurement['snapshot_bytes']==27368902 and measurement['command_count']==15230
snapshot=data('observer-corrected/snapshot.json');assert sha(snapshot)==measurement['snapshot_sha256']==timing['snapshot_sha256']
assert data('observer-first/snapshot.json')==snapshot
assert controls['terminal'] and controls['exit']==0 and controls['timeout']==60 and controls['all_guards_match'] and controls['source_before']==controls['source_after']
for name,h in controls['source_before'].items():assert rows['observer-corrected/'+name]['sha256']==h
assert b'Ran 9 tests' in data('observer-corrected/tests.stderr') and data('observer-corrected/tests.stderr').endswith(b'OK\n')
assert b'FAILED (failures=1)' in data('observer-first/tests.stderr')
paths=[n for n in rows if n.startswith('observer-corrected/') and n.endswith('/actual/cohort-journal.jsonl')];assert len(paths)==1
records=[json.loads(line) for line in data(paths[0]).splitlines()];state=None;commands=[];updates=[]
for i,event in enumerate(records,1):
 assert event['sequence']==i
 if event['kind']=='metadata':state=event['value'];assert not state['terminal'] and state['commands']==[]
 else:
  assert event['kind']=='command';index=event['number'];row=event['value'];assert index==row['number']
  if index==len(commands)+1:commands.append(row)
  else:assert index==len(commands);commands[-1]=row
  if 'observer_control' in row:updates.append(row['observer_control'])
assert updates==list(range(5)) and len(commands)==15230
assert commands[-1].pop('observer_control')==4
state['commands']=commands;assert (json.dumps(state,indent=2)+'\n').encode()==snapshot
assert len(timing['save_seconds'])==len(measurement['trials'])==5
print('PASS %d records; exact failed projection and three-file reduction, eight C-seed/source controls, typed fixture with unchanged assertions, nine offline journal controls and independent15230-command replay. No compiler repair, observer replacement, full foundation/five-route or release acceptance.'%len(rows))
