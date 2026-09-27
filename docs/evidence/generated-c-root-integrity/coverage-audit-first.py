from pathlib import Path
import json,re
base=Path('/home/jkh/nanolang-qualification/vm-effects-20260921/nanolang-record-generated-complete-seal')
reports=base/'reports'
linux=reports/'nanolang-record-generated-linux-complete-seal/nanolang-record-generated-75979-linux'
darwin=reports/'nanolang-record-generated-darwin-recovery-seal'
phases=[(linux,p) for p in ['ordinary','sanitizer','clang-sanitizer']]+[(darwin/'nanolang-record-generated-75979-recovery-puck-backup','ordinary')]+[(darwin/'nanolang-record-generated-75979-sdk-recovery-puck-backup',p) for p in ['clang-ordinary','sanitizer']]
def read(p):return json.loads(p.read_text())
def good(x):return x['returncode']==0 and not x['timeout'] and x['leader_reaped'] and x['group_disappeared'] and not x['errors']
results=[]
for root,phase in phases:
 assert good(read(root/(phase+'-terminal.json')))
 for kind in ['source','tools']:assert read(root/(phase+'-'+kind+'-before.json'))==read(root/(phase+'-'+kind+'-after.json'))
 raw=read(root/(phase+'-artifacts.json'));art={Path(k).name:v for k,v in raw.items()}
 def artifact(name):return read(base/'objects'/art[name]['sha256'])
 statuses=[k for k in art if k.endswith('-status.json')]
 for k in statuses:assert good(artifact(k)),k
 for opt in ['O0','O2']:
  for mode in ['linked','observed']:
   for i in range(73):
    for step in ['build','run']:assert good(artifact(f'{opt}-{mode}-{i:04d}-{step}-status.json'))
 total=0
 for opt in ['O0','O2']:
  for i in range(73):
   prefix=f'{opt}-observed-{i:04d}'
   cov=artifact(prefix+'-fault-coverage.json');plan=artifact(prefix+'-fault-plan.json')
   assert cov['complete'] and cov['product']==prefix
   assert cov['modes']==['one-shot','persistent'] and cov['allocation_calls']==plan['allocation_calls']
   assert cov['positions']==plan['positions']
   end=0;recoveries=0
   for worker in cov['workers']:
    assert worker['begin']==end and worker['end']>end and worker['modes']==2
    assert worker['recoveries']==2*(worker['end']-end)
    assert good(artifact(worker['name']+'-status.json'))
    end=worker['end'];recoveries+=worker['recoveries']
   assert end==cov['allocation_calls'] and recoveries==cov['recoveries']
   total+=recoveries
 results.append({'report_root':str(root.relative_to(reports)),'phase':phase,'child_statuses':len(statuses),'products':292,'fault_coverage_records':146,'recoveries':total,'source_tools_unchanged':True})
result={'scope':'Independent six final generated-C configuration terminal, child status, 73-case O0/O2 linked/observed products and complete fault-range/recovery audit. Current-main integration and production correspondence remain separate.','phases':results,'passed':True}
Path('/home/jkh/nanolang-qualification/generated-c-root-coverage-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
