"""I verify complete handler-flow evidence and replay criteria without products."""
from pathlib import Path
import ast,hashlib,json,random,struct,sys,yaml
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'36adf522f9ad9c54bad67528ed0aa0736db6eb897c837abe9df1a8ad06a22918')
pm,pr,pd,pj=read_archive(base.parent/'constant-reachability-five-linux-20261126','dac03795b503f7531ccf54f285a329ef6875840ffe5097e0ba9950cfff432b73')
q=js('qualified/qualification-result.json');old=pj('qualified/qualification-result.json');s=js('qualified/preparation.json')
assert m['component_only'] and m['pure_nanolang'] and m['full_five_routes']
assert not any(m[k] for k in ['new_bootstrap','profile_admission','body_emission','runtime_emission_complete','pure_cutover','paired','whole_release'])
assert m['source_commit']==pm['source_commit']=='04cee11a861af22dfd4540f369eb2c6bb42a9cfd'
assert q['terminal'] and q['accepted'] and q['all_guards_match'] and q['tests_pass']
assert (q['methods'],q['failures'],q['errors'],q['skipped'])==(2,0,0,0)
assert (q['ordinary_seconds'],q['build_seconds'],q['run_seconds'])==(10,600,60)
assert not q['other_jobs'] and not q['unexpected_sanitizer'] and q['free_bytes']>15*1024**3 and q['filesystem']!='nfs'
assert all(r['cpu']>=1 and r['share']>=.75 for r in q['cpu_rounds'])
assert not q['owned_cleanup']['remaining'] and not q['owned_cleanup']['records']
assert q['source_before']==q['source_after']==s['source_before']
new={'src_nano/nvm2c/effect_flow.nano','tests/nvm2c_effect_flow_driver.nano','tests/nvm2c_effect_flow_reference.c','tests/test_nanolang_translator_effect_flow.py'}
assert set(q['source_after'])==set(old['source_after'])|new
for name,h in q['source_after'].items():
 assert rows['qualified/source/'+name]['sha256']==h
 if name not in new|{'Makefile.gnu'}:assert old['source_after'][name]==h
block=b'.PHONY: test-nanolang-translator-effect-flow\ntest-nanolang-translator-effect-flow: $(COMPILER_C) nano_virt nano_vm nvm2c nvm2c-runtime\n\tpython3 -m unittest -v tests.test_nanolang_translator_effect_flow\n\ntest-units: test-nanolang-translator-effect-flow\n\n'
make=data('qualified/source/Makefile.gnu');assert make.count(block)==1 and make.replace(block,b'',1)==pd('qualified/source/Makefile.gnu')
assert q['images_before']==q['images_after']==old['images_after'] and len(q['images_after'])==12
for name,h in q['images_after'].items():assert rows['qualified/source/'+name]['sha256']==h
assert q['host_tools_before']==q['host_tools_after']==old['host_tools_after'] and len(q['host_tools_after'])==22
assert q['producer_before']==q['producer_after']==old['producer_after']=='f28b2bf49e94b5033e6b579b07f5663f0fc7151f1a933ab09efef53e0e13cd74'
assert q['producer_hosts_before']==q['producer_hosts_after']==old['producer_hosts_after'] and len(q['producer_hosts_after'])==6
assert q['producer_receipt_before']==q['producer_receipt_after']==old['producer_receipt_after']
for path,h in {**q['host_tools_after'],**q['producer_hosts_after'],q['producer']:q['producer_after']}.items():assert rows['external/'+path.lstrip('/')]['sha256']==h
assert len(q['commands'])>20
for n,c in enumerate(q['commands'],1):
 assert c['number']==n and c['streams_complete'] and c['exit']==0
 assert c['effective_budget']==(600 if c['build'] else 60) and c['timeout_requested']==60
 assert not c['cleanup']['remaining'] and not c['cleanup']['records']
 for stream in ['stdout','stderr']:assert rows['qualified/command-%03d.%s'%(n,stream)]['sha256']==c[stream+'_sha256']
 for a in c['artifacts']:
  if a.get('kind')!='symlink':assert rows['qualified/retained/'+a['sha256']]['bytes']==a['bytes'] and rows['qualified/retained/'+a['sha256']]['sha256']==a['sha256']
namespace={};exec(compile(data('collector-source/replay.py'),'replay.py','exec'),namespace)
r=namespace['replay'](data);assert r['passed'] and r['methods']==2 and r['product_commands_executed']==0
label='failed-call-expectation';fq=js(label+'/qualification-result.json');fs=js(label+'/preparation.json')
assert fq['terminal'] and not fq['accepted'] and fq['all_guards_match']
assert (fq['methods'],fq['failures'],fq['errors'],fq['skipped'])==(2,1,0,0)
assert not fq['owned_cleanup']['remaining'] and not fq['owned_cleanup']['records']
assert fq['source_before']==fq['source_after']==fs['source_before']
for name,h in fq['source_after'].items():
 assert rows[label+'/source/'+name]['sha256']==h
 if name!='tests/test_nanolang_translator_effect_flow.py':assert q['source_after'][name]==h
assert fq['images_before']==fq['images_after']==q['images_after'] and fq['host_tools_before']==fq['host_tools_after']==q['host_tools_after']
assert fq['producer_hosts_before']==fq['producer_hosts_after']==q['producer_hosts_after'] and fq['producer_before']==fq['producer_after']==q['producer_after']
fr=namespace['replay'](lambda path:data(label+'/'+path.removeprefix('qualified/')),expected_pass=False)
assert not fr['passed'] and fr['methods']==2 and fr['product_commands_executed']==0
storage=js('storage-refused/qualification-result.json');assert storage['terminal'] and not storage['accepted'] and storage['all_guards_match'] and not storage['commands']
assert storage['filesystem']!='nfs' and storage['free_bytes']<=15*1024**3
for name,h in storage['source_after'].items():
 assert rows['storage-refused/source/'+name]['sha256']==h
 assert fq['source_after'][name]==h
assert not storage['owned_cleanup']['remaining'] and not storage['owned_cleanup']['records']
# I regenerate fixture bytes/expectations with pure Python, preserving the old
# packets and correcting only the two mistaken direct-call expectations.
ops={v['name']:v for v in yaml.safe_load(data('qualified/source/spec/nanoisa.yaml'))['legacy_opcodes']}
def fixtures(raw):
 tree=ast.parse(raw);ns=dict(struct=struct,random=random,OPS=ops,FMT={'U8':'B','U16':'H','U32':'I','I32':'i','I64':'q','F64':'Q'})
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'fixture-functions','exec'),ns)
 return ns['cases']()
before=fixtures(data(label+'/source/tests/test_nanolang_translator_effect_flow.py'))
after=fixtures(data('qualified/source/tests/test_nanolang_translator_effect_flow.py'))
assert len(after)==len(before)+2
newrows=[v for v in after if v[0].startswith('out-of-range-other-')];assert len(newrows)==2 and all(v[2]=='0' for v in newrows)
kept=[v for v in after if not v[0].startswith('out-of-range-other-')]
changes=[]
for previous,current in zip(before,kept):
 # Metadata labels include their row number; adding neighbors shifts those
 # labels but never changes their packet bytes or expected result.
 assert previous[1]==current[1]
 if previous[2]!=current[2]:
  assert previous[0] in ('bad-other-3b01000000','bad-other-3301000000') and previous[2]=='0' and current[2]=='1:0'
  changes.append(previous[0])
assert len(changes)==2
assert js('collector-source/live-refusal.json')==dict(exit=1,output_created=False)
print(json.dumps(dict(accepted=True,records=len(rows),manifest=sha((base/'manifest.json').read_bytes()),matrix=len(after),explicit_expectations=sum(v[2] is not None for v in after),replay=r,failed_replay=fr,product_commands_executed=0)))
