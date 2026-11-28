"""I verify the complete isolated six-leaf gate, not product cutover."""
from pathlib import Path
import hashlib,json,subprocess,sys,shlex
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,records,data,js=read_archive(base,'0b2c23e57700f648eb71eca18b802888251fefe74585d14a2b52635591052b7c')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
q=js('comparison/cohort-result.json');s=js('comparison/result.json')
assert not m['accepted_component'] and not m['full_five_routes'] and m['working_overlay'] and not m['whole_release']
assert m['source_commit']==q['source_pin']==s['source_base']=='67d99e5121521428740d385d10071b9679e66013'
assert q['terminal'] and not q.get('accepted_component',False) and q['all_guards_match'] and not q['whole_release']
assert q['tests']==dict(tests=6,failures=4,errors=0,skips=0,passed=False)
assert q['original_frame_cases']==1001 and q['new_cases']==dict(owners=14542,aggregates=15222,effects=8128,value_calls=15644,call_context=834,analysis_cache=3317)
root=Path(m['roots']['comparison']);source=root/'source'
def mapped(name):
 p=Path(name)
 return 'comparison/'+str(p.relative_to(root)) if p.is_relative_to(root) else 'external/'+str(p).lstrip('/')
for group in ['source','images','tools','hosts']:
 assert q[group+'_before']==q[group+'_after']==s[group+'_before']==s[group+'_after']
 for n,h in q[group+'_before'].items():assert records['comparison/source/'+n if group in ['source','images'] else mapped(n)]['sha256']==h,n
assert q['object_inputs']==q['objects_after']
for n,h in q['object_inputs'].items():assert records[mapped(n)]['sha256']==h
assert q['observer_before']==q['observer_after']
for n,h in q['observer_before'].items():assert records[mapped(n)]['sha256']==h
assert q['producer_after']==q['producer_sha256']==s['producer_sha256']=='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
assert q['producer_source_pin']==s['bootstrap_source_pin']=='7b9f08ee0941c23810d18fa3f8ebfbb759c968de'
assert records[mapped(s['producer'])]['sha256']==q['producer_sha256']
origin=js('origin/nanoc_bootstrap.json');assert origin['phase']=='stage2' and origin['raw_stage1_stage2_equal'] and origin['artifacts']['nanoc_stage2']==q['producer_sha256']
parent=js('query-dependency/cohort-result.json');assert parent['terminal'] and parent['accepted_component'] and parent['all_guards_match'] and parent['source_before']==q['source_before'] and parent['overlay_sha256']==q['overlay_sha256']==s['overlays']
assert q['query_evidence']==s['query_evidence'] and s['query_evidence']['result_sha256']==sha(data('query-dependency/cohort-result.json'))
assert s['query_evidence']['commit']=='dec468a6bda61adf5d268c365832811c448d69f0' and s['query_evidence']['manifest_sha256']=='8de5afcc2d1cd1f3428618fa22ad5d1d3da101ebcbfcb01b3057020fbe02c709'
admit=js('comparison/admission-result.json');assert admit['terminal'] and admit['accepted'] and not admit['other_jobs'] and admit['free_bytes']>15*1024**3 and admit['filesystem']!='nfs'
assert all(r['share']>=.75 for r in s['cpu_rounds']) and s['ordinary_seconds']==q['ordinary_seconds']==10
for name in ['CMD_TIMEOUT','NANO_SHADOW_TIMEOUT_SECONDS','NANO_CC','NANO_CFLAGS','NANO_LDFLAGS','NANOLANG_TRANSLATOR_SOURCE_COMPILER','NANOLANG_TRANSLATOR_SOURCE_MODULE']:assert s['environment_before'][name] is None
assert len(q['commands'])==11093 and [c['number'] for c in q['commands'] if c['exit']!=0]==[11093]
assert all(not c.get('interrupted_or_timeout') for c in q['commands'])
assert b'I require the matching scalar shape at this array write' in data('comparison/command-11093.stderr')
assert sum(c['build'] for c in q['commands'])==41 and len(q['providers'])==12
for label,key in [('global-array','global_array'),('foundation','foundation')]:
 parent=js(label+'-dependency/cohort-result.json');assert parent['terminal'] and parent['accepted_component'] and parent['all_guards_match']
 assert sha(data(label+'-dependency/cohort-result.json'))==s[key+'_evidence']['result_sha256']
 assert q[key+'_evidence']==s[key+'_evidence'] and parent['producer_sha256']==q['producer_sha256']
 if key=='global_array':assert parent['source_before']==q['source_before'] and parent['overlay_sha256']==q['overlay_sha256']
# I replay every command/metadata event into the exact terminal JSON.
state=None;history=[];committed=False
for i,line in enumerate(data('comparison/cohort-journal.jsonl').splitlines(),1):
 e=json.loads(line);assert e['sequence']==i and not committed
 if e['kind']=='metadata':state=e['value'];assert state['commands']==[]
 elif e['kind']=='command':
  n=e['number'];assert n==e['value']['number'] and n in (len(history),len(history)+1) and n>0
  if n==len(history)+1:history.append(e['value'])
  else:history[-1]=e['value']
 elif e['kind']=='commit':
  assert state['terminal'] and e['commands']==len(history);state['commands']=history;raw=(json.dumps(state,indent=2)+'\n').encode()
  assert len(raw)==e['bytes'] and sha(raw)==e['sha256'] and raw==data('comparison/cohort-result.json');committed=True
 else:raise AssertionError('I require a known journal event')
assert committed and state==q
# I retain exact imported module/provider bindings before each translation.
for index,(name,r) in enumerate(q['providers'].items()):
 assert r['read_only'] and r['before_translation'] and r['module_sha256']==r['module_after']==records[mapped(name)]['sha256']
 assert r['libraries_before']==r['libraries_after']
 for n,h in r['libraries_before'].items():assert records[mapped(n)]['sha256']==h
 lines=data('comparison/provider-%d.stdout'%index).decode().splitlines()
 assert [shlex.split(line) for line in lines if line.startswith('.import ')]==r['bindings']
def inspect_case(group,new,case,line):
 expected=case['expected']
 if isinstance(expected,str):
  if group=='original' and not case['name'].startswith(('global-algebra-','bounded-fifo-')):assert line.startswith(expected)
  else:assert line==expected
 else:
  pieces=line.split('~') if new=='owners' else [line];wants=expected if new=='owners' else [expected];assert len(pieces)==len(wants)
  for text,want in zip(pieces,wants):
   chunks=text.split('|');valid,message,frame_,authority=chunks[:4];local_,stack,relations,context=frame_.split('/')
   assert valid==str(int(want['valid'])) and message==want['message'] and stack==want['stack'] and authority==want['authority']
   assert relations==(want['globals'] if new=='calls' else '9,') and context==(want['context'] if new in ['effects','calls','context','cache'] else '1,23,5,7,')
   assert len(chunks)==(7 if new=='calls' else 5 if new in ['effects','context','cache'] else 4)
   if new=='calls':assert chunks[4]==want['allowed'] and chunks[5]==want['keys'] and chunks[6]==str(int(want['derived']))
   elif new in ['effects','context','cache']:assert chunks[4]==want['keys']

# I distinguish retained completed reports from full component acceptance.
manifests=[n for n in records if n.startswith('comparison/tmp/nano-affine-') and n.endswith('/manifest.json')]
assert len(manifests)==6
counts={'owners':14542,'aggregates':15222,'effects':8128,'calls':15644,'context':834,'cache':3317}
leaves={'owners':'owners','aggregates':'aggregates','effects':'effects','calls':'value_calls','context':'call_context','cache':'analysis_cache'}
report_hashes={'owners':'43cf511feec513c6ead8bde06620079cff0be6c4f0db5e8154db153beea271c2','aggregates':'f26b2400f0ec919b5fcf5c15a869f14db43bfd2a6394842ac5b6040a9777a468','effects':'0521f0169aa763ebe53c45311cb89ff343c42589d22deea3b3cac3f5a257cba9','calls':'9e7178f7a0944609b3486cd53cb5282cab14ebdf7df45c1fec','context':'b0676f47e95f91952daab6a64008e14ec2342e67f5b60592f201c5f6ba331989'}
# I use the full independently audited value-call digest.
report_hashes['calls']='9e7178f7a0944609b3486cd53cb528232aa8cf5125282cab14ebdf7df45c1fec'
seen=set();report_commands=set();total=0;bad_seen=set()
for name in manifests:
 doc=js(name);folder=name.rsplit('/',1)[0];new=({c['group'] for c in doc['cases']}-{'original'}).pop();assert new in counts and new not in seen;seen.add(new)
 leaf=leaves[new];physical=root/folder.removeprefix('comparison/')
 assert len(doc['cases'])==1001+counts[new]
 for case in doc['cases']:assert records[mapped(case['path'])]['sha256']==case['sha256']
 assert len(doc['routes'])==(12 if new in ['owners','context'] else 0 if new=='cache' else 2)
 prefix={'actual-c':[q['reference_argv'][leaf][q['reference_argv'][leaf].index('-o')+1]],'c-seed-native':[str(physical/'decoder-native')],
 'c-producer-nanoisa-vm':[str(source/'bin/nano_vm'),str(physical/'decoder.nvm'),'--'],
 'c-producer-sanitized-native':[str(physical/'decoder-aot')],
 'nanolang-producer-nanoisa-vm':[str(source/'bin/nano_vm'),str(physical/'source-decoder.nvm'),'--'],
 'nanolang-producer-sanitized-native':[str(physical/'source-decoder-aot')]}
 for group,count in [('original',1001),(new,counts[new])]:
  cases=[c for c in doc['cases'] if c['group']==group];paths={c['path'] for c in cases}
  if new=='cache':continue
  baseline=data(folder+'/'+group+'-actual-c.reports');lines=baseline.decode().splitlines();assert len(lines)==count
  assert sha(baseline)==('f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0' if group=='original' else report_hashes[new])
  for case,line in zip(cases,lines):inspect_case(group,new,case,line)
  streams=[n for n in records if n.startswith(folder+'/'+group+'-') and n.endswith('.reports')]
  for stream in streams:
   label=stream[len(folder+'/'+group+'-'):-len('.reports')];assert label in prefix
   raw=data(stream)
   if group=='original' and label=='c-seed-native' and new in ['aggregates','effects','calls']:
    a=baseline.splitlines();b=raw.splitlines();assert len(a)==len(b)==1001
    diffs=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    assert diffs==list(range(336,1001)) and all(b[i].startswith(a[i]+b'|') for i in diffs);bad_seen.add(new)
   else:assert raw==baseline
   commands=[c for c in q['commands'] if c['argv'][:len(prefix[label])]==prefix[label] and len(c['argv'])>len(prefix[label]) and c['argv'][len(prefix[label])] in paths]
   assert [n for c in commands for n in c['argv'][len(prefix[label]):]]==[c['path'] for c in cases]
   assert b''.join(data('comparison/command-%04d.stdout'%c['number']) for c in commands)==raw
   for c in commands:
    inputs=c['argv'][len(prefix[label]):];assert c['timeout']==(600 if group=='original' else 60)
    assert len(inputs)<=(8 if group=='original' else 32)
    if group!='original':assert len(inputs)==1 or sum(records[mapped(n)]['bytes'] for n in inputs)<=8192
    assert c['number'] not in report_commands;report_commands.add(c['number'])
   total+=count
assert seen==set(counts) and bad_seen=={'aggregates','effects','calls'}
for c in q['commands']:
 if c['number'] not in report_commands:assert c['timeout']==(600 if c['build'] else 60)
assert len(report_commands)+41+6==len(q['commands'])
print(json.dumps(dict(terminal=True,accepted_failure_archive=True,accepted_component=False,whole_release=False,commands=len(q['commands']),retained_reports=total,records=len(records))))
