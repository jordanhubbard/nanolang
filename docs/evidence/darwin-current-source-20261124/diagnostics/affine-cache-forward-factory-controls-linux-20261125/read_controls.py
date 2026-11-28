"""I verify my bounded forward-factory construction controls."""
from pathlib import Path
import json,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,records,data,js=read_archive(base,'90e8f738272ecf7f2ec36730250981d56a46ff5e8ccc5f56800c8aafc8474cd6')
assert m['accepted_minimal_factory_controls'] and not m['accepted_component'] and not m['whole_release']
q=js('controls/result.json');v=js('controls/native-result.json');draft=js('draft/result.json')
assert q['terminal'] and q['all_guards_match'] and len(q['commands'])==12
assert v['terminal'] and v['all_guards_match'] and v['accepted_native_shadow_controls'] and len(v['commands'])==4
assert q['ordinary_seconds']==v['ordinary_seconds']==10
for state in [q,v]:
 assert state['before']==state['after']
 for n,h in state['before'].items():assert records['external/'+n.lstrip('/')]['sha256']==h
for c in q['commands']:
 assert c['timeout']==(60 if c['label'].endswith('-vm') else 600)
 assert c['exit']==(1 if c['label']=='forward-genuine-translate' else 0)
 if c['label'].endswith('-vm'):assert data('controls/'+c['label']+'.stdout')==b'3\n'
assert b'I require the matching scalar shape at this array write' in data('controls/forward-genuine-translate.stderr')
for c in v['commands']:
 assert c['exit']==0 and c['timeout']==(600 if c['label'].endswith('-build') else 60)
 if c['label'].endswith('-run'):assert data('controls/'+c['label']+'.stdout')==b'3\n'
forward=data('controls/forward.nano');inline=data('controls/inline.nano')
needle=b'    let input: Cache = (make_cache 1)'
replacement=b'    let initial_row: array<int> = [1]\n    let mut initial_keys: array<array<int>> = []\n    set initial_keys (array_push initial_keys initial_row)\n    let input: Cache = Cache { keys: initial_keys }'
assert forward.count(needle)==1 and forward.replace(needle,replacement)==inline
assert b'shadow make_cache' in forward and b'shadow exercise' in forward
assert draft['terminal'] and draft['all_guards_match']
wrong=[c for c in draft['commands'] if c['label'].endswith('-c-compile')];assert len(wrong)==2 and all(c['exit']==1 for c in wrong)
for c in wrong:assert b'I currently accept C-seed --emit-nvm only for checked File graphs.' in data('draft/'+c['label']+'.stderr')
print(json.dumps(dict(terminal=True,accepted_minimal_factory_controls=True,commands=16,expected_translation_refusals=1,product_acceptance=False,whole_release=False)))
