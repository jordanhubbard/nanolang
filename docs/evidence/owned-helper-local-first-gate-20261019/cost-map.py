from pathlib import Path
import json,sys,collections,copy
root=Path('/Users/jkh/Src/nanolang-owned-nested-depth2');sys.path.insert(0,str(root/'scripts'))
from nanoisa_owned_reconstruction import OwnedAnalyze,OwnedEmit
from nanoisa_owned_rendering import plan_reassignment_nano
out=Path(__file__).parent;facts=json.loads((out/'nano_virt.facts.json').read_text());fns=[OwnedAnalyze(facts,i).run() for i in range(2)];em=OwnedEmit(fns,'nano',facts['layouts'])
rows=[]
for i,fn in enumerate(fns):
 sums=collections.Counter();counts=collections.Counter()
 for st in em.walk_statements(fn.body):
  kind=st[0];counts[kind]+=1
  unit=copy.copy(fn);unit.locals={};unit.params=[];unit.body=[] if kind in ('if','while','while_observe') else [st]
  key='inline_add' if kind=='let' and st[2].kind=='arithmetic' and st[2].value=='add' else kind
  sums[key]+=em.nano_local_cost(unit)
 row={'function':i,'original_input_locals':facts['functions'][i]['locals'],'scalar_declarations':len(fn.locals),'params':len(fn.params),'statement_counts':dict(counts),'costs':dict(sums),'total':em.nano_local_cost(fn)}
 # These are diagnostic cloned-IR experiments with EXISTING transformations,
 # not ordinary CLI admission or actual regenerated local counts.
 folded=copy.deepcopy(fn);folded.body=em.fold_adjacent_assertions(folded.body)
 row['adjacent_bool_assert_cost']=em.nano_local_cost(folded)
 planned,receipt=plan_reassignment_nano(fn);row['existing_plan_receipt']=receipt;row['existing_plan_cost']=em.nano_local_cost(planned)
 combined,receipt=plan_reassignment_nano(folded);row['combined_plan_receipt']=receipt;row['combined_cost']=em.nano_local_cost(combined)
 if any(st[0]=='let' and st[2].kind=='arithmetic' and st[2].value=='add' for st in em.walk_statements(combined.body)):
  em.scratch_functions.add(id(combined));row['combined_private_add_bank_cost']=em.nano_local_cost(combined)
 rows.append(row)
(out/'cost-map.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
