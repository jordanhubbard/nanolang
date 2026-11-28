"""I verify the observer bound without promoting three routes to native acceptance."""
from pathlib import Path
import ast,contextlib,hashlib,io,json,random,re,runpy,struct,sys,yaml
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda raw:hashlib.sha256(raw).hexdigest()
prior=base.parent/'callables-three-route-diagnostic-linux-20261126'
assert sha((prior/'manifest.json').read_bytes())=='de324b36e969f672e9fe49074b4166b16284c789a102415191d40c60b9a8694b'
assert sha((prior/'read_archive.py').read_bytes())=='98c1ceae8867cf9c565bb24507107c266bbc0b1460d1d3e6410255a25b4b9846'
with contextlib.redirect_stdout(io.StringIO()) as stream:previous=runpy.run_path(str(prior/'read_archive.py'))
prior_result=json.loads(stream.getvalue());assert prior_result['evidence_verified'] and not prior_result['component_accepted'] and prior_result['product_commands_executed']==0
old=previous['q'];pd=previous['data'];m,rows,data,js=read_archive(base,'83e3c9eb5b6c51c935524d6b56494268b7ea9426f4a00551caf36e58352c4e59');q=js('qualified/qualification-result.json');s=js('qualified/preparation.json')
assert m['component_only'] and m['pure_nanolang'] and m['diagnostic_only']
assert not any(m[k] for k in ['full_five_routes','native_acceptance','component_accepted','new_bootstrap','profile_admission','body_emission','runtime_emission_complete','pure_cutover','paired','whole_release'])
assert q['terminal'] and q['diagnostic_pass'] and q['tests_pass'] and q['all_guards_match']
assert not q['accepted'] and not q['full_five_routes'] and q['diagnostic_only'] and q['acceptance_withheld']
assert (q['methods'],q['failures'],q['errors'],q['skipped'])==(3,0,0,0)
assert (q['ordinary_seconds'],q['build_seconds'],q['run_seconds'])==(10,600,60)
assert q['source_before']==q['source_after']==s['source_before'] and set(q['source_after'])==set(old['source_after']) and len(q['source_after'])==4002
changed={'tests/nvm2c_callables_reference.c','tests/test_nanolang_translator_callables.py'}
for name,h in q['source_after'].items():
 assert rows['qualified/source/'+name]['sha256']==h
 if name not in changed:assert old['source_after'][name]==h
source=pd('qualified/source/tests/nvm2c_callables_reference.c')
needle=b'if(pos<=(size_t)size && ic<=((size_t)size-pos)/20){'
replacement=b'if(pos<=(size_t)size && ic<=((size_t)size-pos)/20 && sc<=((size_t)size-pos)/4){'
assert source.count(needle)==1 and source.replace(needle,replacement,1)==data('qualified/source/tests/nvm2c_callables_reference.c')
assert q['images_before']==q['images_after']==old['images_after'] and len(q['images_after'])==49
for name,h in q['images_after'].items():assert rows['qualified/source/'+name]['sha256']==h
assert q['host_tools_before']==q['host_tools_after']==old['host_tools_after'] and len(q['host_tools_after'])==22
assert q['producer_before']==q['producer_after']==old['producer_after']
assert q['producer_hosts_before']==q['producer_hosts_after']==old['producer_hosts_after'] and len(q['producer_hosts_after'])==6
assert q['producer_receipt_before']==q['producer_receipt_after']==old['producer_receipt_after']
for name,h in {**q['host_tools_after'],**q['producer_hosts_after'],q['producer']:q['producer_after']}.items():assert rows['external/'+name.lstrip('/')]['sha256']==h
assert not q['owned_cleanup']['remaining'] and not q['owned_cleanup']['records']
assert not q['other_jobs'] and not q['unexpected_sanitizer'] and q['filesystem']!='nfs' and q['free_bytes']>15*1024**3
assert all(r['cpu']>=1 and r['share']>=.75 for r in q['cpu_rounds'])
for number,command in enumerate(q['commands'],1):
 assert command['number']==number and command['exit']==0 and command['streams_complete']
 assert command['effective_budget']==(600 if command['build'] else 60) and command['timeout_requested']==60
 assert not command['cleanup']['remaining'] and not command['cleanup']['records']
 for stream_name in ['stdout','stderr']:assert rows['qualified/command-%03d.%s'%(number,stream_name)]['sha256']==command[stream_name+'_sha256']
 for item in command['artifacts']:
  if item.get('kind')!='symlink':assert rows['qualified/retained/'+item['sha256']]['sha256']==item['sha256'] and rows['qualified/retained/'+item['sha256']]['bytes']==item['bytes']
namespace={};exec(compile(data('collector-source/replay.py'),'replay.py','exec'),namespace)
r=namespace['replay'](data);assert r['passed'] and r['methods']==3 and r['diagnostic_only'] and not r['component_accepted'] and r['product_commands_executed']==0
ops={v['name']:v for v in yaml.safe_load(data('qualified/source/spec/nanoisa.yaml'))['legacy_opcodes']}
def fixtures(raw):
 tree=ast.parse(raw);scope=dict(struct=struct,random=random,re=re,OPS=ops,FMT={'U8':'B','U16':'H','U32':'I','I32':'i','I64':'q','F64':'Q'})
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'pure-fixture-functions','exec'),scope)
 return scope['cases']()
after=fixtures(data('qualified/source/tests/test_nanolang_translator_callables.py'));before=previous['after'];assert after[:len(before)]==before and len(after)==len(before)+11
assert all(row[2]==('packet',) for row in after[len(before):])
command=next(c for c in q['commands'] if c['argv'][0].endswith('/original-callables'))
assert b'505 passed, 0 failed' in data('qualified/command-%03d.stdout'%command['number'])
item=next(a for a in command['artifacts'] if a['path'].endswith('/original.packets'))
assert data('qualified/retained/'+item['sha256'])==previous['original']
controls=js('collector-source/collector-controls.json');assert len(controls)==2
for control in controls:
 assert control['exit']==1 and not control['output_created'] and control['collector_sha256']==sha(data('collector-source/package.py'))
 assert b'AssertionError' in data('collector-source/'+control['label']+'-refusal.stderr')
control=js('nonterminal-control/qualification-result.json');expected=dict(q);expected['terminal']=False;assert control==expected
print(json.dumps(dict(evidence_verified=True,component_accepted=False,native_acceptance=False,full_five_routes=False,diagnostic_pass=True,
 records=len(rows),prior_records=prior_result['records'],matrix=len(after),new_boundaries=11,original_modules=previous['original_count'],
 replay=r,prior_journals=len(prior_result['replays']),product_commands_executed=0)))
