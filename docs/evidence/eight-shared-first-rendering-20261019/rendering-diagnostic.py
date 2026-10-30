from pathlib import Path
import sys,json,copy,collections
w=Path('/Users/jkh/Src/nanolang-51-owned-float-design');sys.path.insert(0,str(w/'scripts'))
from nanoisa_owned_reconstruction import OwnedAnalyze,OwnedEmit
from nanoisa_owned_rendering import plan_reassignment_nano
out=Path(__file__).parent;m=json.loads((out/'nano_virt-changed-facts.json').read_text());f=[OwnedAnalyze(m,i).run() for i in range(2)];e=OwnedEmit(f,'nano',m['layouts']);before=repr(f);rows=[]
for index,fn in enumerate(f):
 planned,receipt=plan_reassignment_nano(fn)
 row={'index':index,'receipt':receipt,'original_cost':e.nano_local_cost(fn),'planned_cost':e.nano_local_cost(planned),'original_counts':dict(collections.Counter(s[0] for s in e.walk_statements(fn.body))),'planned_counts':dict(collections.Counter(s[0] for s in e.walk_statements(planned.body)))}
 e.scratch_functions.add(id(planned));row['planned_existing_scratch_cost']=e.nano_local_cost(planned);e.scratch_functions.clear();rows.append(row)
assert repr(f)==before
(out/'rendering-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
