"""I verify selected cache behavior and read-only refusal localization."""
from pathlib import Path
import hashlib,json,re,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,records,data,js=read_archive(base,'b3bb1af0d14d522da09d155c2ae02313e60f33daa7172a87cbc1937cbba31426')
assert m['accepted_read_only_localization'] and m['working_overlay']
assert not any(m[n] for n in ['accepted_component','whole_release','full_five_routes'])
q=js('diagnosis/detail-result.json');v=js('diagnosis/vm-controls-result.json');parent=js('failed-parent/cohort-result.json')
assert q['terminal'] and q['read_only_debugger'] and not q['product_acceptance']
assert q['function']==8 and q['offset']==1645 and q['receiver_kind']=='UNK' and q['value_kind']=='ARR'
assert not q['final_pass'] and not q['canonical_records']
assert v['terminal'] and v['accepted_selected_vm_controls'] and v['all_guards_match'] and not v['product_acceptance']
for state in [q,v]:
 assert state['before']==state['after']
 for n,h in state['before'].items():assert records['external/'+n.lstrip('/')]['sha256']==h
for n,h in q['traces'].items():assert records['diagnosis/'+n]['sha256']==h
assert len(v['commands'])==27 and len(v['cases'])==9
for i,c in enumerate(v['commands']):
 assert c['exit']==0 and c['timeout']==60
 case=v['cases'][i%9];assert c['argv'][-1]==case['path']
 assert records['external/'+case['path'].lstrip('/')]['sha256']==case['sha256']
 assert data('diagnosis/'+c['label']+'.stdout').decode()==case['expected']+'\n'
rows=js('diagnosis/function8-decoded.json');at={r['offset']:r for r in rows}
assert [(at[n]['name'],at[n]['operands']) for n in [1633,1636,1639,1642,1645]]==[('LOAD_LOCAL',[27]),('AGG_GET',[5]),('LOAD_LOCAL',[34]),('LOAD_LOCAL',[42]),('ARR_SET',[])]
assert at[1065]['name']=='CALL' and at[1065]['operands']==[37] and at[1070]['operands']==[27]
trace=data('diagnosis/trace-opcode.stdout').decode()
for text in ['idx = 8','val.kind = 3','arr.kind = 2','facts->final = 0','$1 = false',"'offset': 1645"]:assert text in trace
functions=re.findall(r'^\.function (\S+)',data('failed-parent/provider-11.stdout').decode(),re.M)
assert functions[8]=='describe_cache' and functions[37]=='__nano_module_15_create'
c_functions=re.findall(r'^\.function (\S+)',data('failed-parent/provider-10.stdout').decode(),re.M)
assert c_functions.index('describe_cache')==162
assert parent['terminal'] and parent['tests']['failures']==4 and parent['all_guards_match']
assert parent['commands'][-1]['number']==11093 and parent['commands'][-1]['exit']==1
assert b'I require the matching scalar shape at this array write' in data('failed-parent/command-11093.stderr')
for n,h in q['before'].items():
 if n.endswith('/src/nanoisa/nvm2c.c'):assert parent['source_before']['src/nanoisa/nvm2c.c']==h
 if n.endswith('/source-decoder.nvm'):assert parent['providers'][n]['module_sha256']==h
print(json.dumps(dict(terminal=True,accepted_read_only_localization=True,selected_cases=9,commands=27,function=8,offset=1645,product_acceptance=False,whole_release=False)))
