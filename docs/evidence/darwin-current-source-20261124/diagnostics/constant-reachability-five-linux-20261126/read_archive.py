"""I verify my full exact-graph observations without executing product commands."""
from pathlib import Path
import hashlib,json,sys,types
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'dac03795b503f7531ccf54f285a329ef6875840ffe5097e0ba9950cfff432b73')
pm,pr,pd,pj=read_archive(base.parent/'shape-solver-five-linux-20261126','aed0f2767d13142d8d58fcc30af21d69a7b948a8caf13b320d1d09e725bc5874')
q=js('qualified/qualification-result.json');old=pj('qualified/qualification-result.json');s=js('qualified/preparation.json')
assert m['component_only'] and m['pure_nanolang'] and m['full_five_routes']
assert not any(m[k] for k in ['new_bootstrap','profile_admission','body_emission','runtime_emission_complete','pure_cutover','paired','whole_release'])
assert m['source_commit']==pm['source_commit']=='04cee11a861af22dfd4540f369eb2c6bb42a9cfd'
assert q['terminal'] and q['accepted'] and q['all_guards_match'] and q['tests_pass']
assert (q['methods'],q['failures'],q['errors'],q['skipped'])==(4,0,0,0)
assert (q['ordinary_seconds'],q['build_seconds'],q['run_seconds'])==(10,600,60)
assert not q['other_jobs'] and not q['unexpected_sanitizer'] and q['free_bytes']>15*1024**3 and q['filesystem']!='nfs'
assert all(r['cpu']>=1 and r['share']>=.75 for r in q['cpu_rounds'])
assert not q['owned_cleanup']['remaining'] and not q['owned_cleanup']['records']
assert q['source_before']==q['source_after']==s['source_before'] and len(q['source_after'])==3993
new={'src_nano/nvm2c/constant_reachability.nano','tests/nvm2c_constant_reachability_driver.nano','tests/nvm2c_constant_reachability_reference.c','tests/test_nanolang_translator_constant_reachability.py'}
assert set(q['source_after'])==set(old['source_after'])|new
for name,h in q['source_after'].items():
 assert rows['qualified/source/'+name]['sha256']==h
 if name not in new|{'Makefile.gnu'}:assert old['source_after'][name]==h
block=b'.PHONY: test-nanolang-translator-constant-reachability\ntest-nanolang-translator-constant-reachability: $(COMPILER_C) nano_virt nano_vm nvm2c nvm2c-runtime\n\tpython3 -m unittest -v tests.test_nanolang_translator_constant_reachability\n\ntest-units: test-nanolang-translator-constant-reachability\n\n'
make=data('qualified/source/Makefile.gnu');assert make.count(block)==1 and make.replace(block,b'',1)==pd('qualified/source/Makefile.gnu')
assert q['images_before']==q['images_after']==old['images_after'] and len(q['images_after'])==12
for name,h in q['images_after'].items():assert rows['qualified/source/'+name]['sha256']==h
assert q['host_tools_before']==q['host_tools_after']==old['host_tools_after'] and len(q['host_tools_after'])==22
assert q['producer_before']==q['producer_after']==old['producer_after']=='f28b2bf49e94b5033e6b579b07f5663f0fc7151f1a933ab09efef53e0e13cd74'
assert q['producer_hosts_before']==q['producer_hosts_after']==old['producer_hosts_after'] and len(q['producer_hosts_after'])==6
assert q['producer_receipt_before']==q['producer_receipt_after']==old['producer_receipt_after']
for path,h in {**q['host_tools_after'],**q['producer_hosts_after'],q['producer']:q['producer_after']}.items():assert rows['external/'+path.lstrip('/')]['sha256']==h
assert len(q['commands'])>500
for n,c in enumerate(q['commands'],1):
 assert c['number']==n and c['streams_complete'] and c['exit']==0
 assert c['effective_budget']==(600 if c['build'] else 60) and c['timeout_requested']==60
 assert not c['cleanup']['remaining'] and not c['cleanup']['records']
 for stream in ['stdout','stderr']:assert rows['qualified/command-%03d.%s'%(n,stream)]['sha256']==c[stream+'_sha256']
 for a in c['artifacts']:
  if a.get('kind')!='symlink':assert rows['qualified/retained/'+a['sha256']]['bytes']==a['bytes'] and rows['qualified/retained/'+a['sha256']]['sha256']==a['sha256']
namespace={};exec(compile(data('collector-source/replay.py'),'replay.py','exec'),namespace)
r=namespace['replay'](data);assert r['passed'] and r['methods']==4 and r['product_commands_executed']==0
failure=js('failed-visit-fixture/qualification-result.json');assert failure['terminal'] and not failure['accepted'] and failure['all_guards_match']
assert (failure['methods'],failure['failures'],failure['errors'])==(4,1,0)
assert not failure['owned_cleanup']['remaining'] and not failure['owned_cleanup']['records']
for name,h in q['source_after'].items():
 if name not in {'tests/test_nanolang_translator_constant_reachability.py','src_nano/nvm2c/constant_reachability.nano'}:assert failure['source_after'][name]==h
assert b"'1' != '0'" in data('failed-visit-fixture/qualification.stderr')
for label in ['failed-visit-fixture','failed-permutation']:
 fq=js(label+'/qualification-result.json');fs=js(label+'/preparation.json')
 assert fq['source_before']==fq['source_after']==fs['source_before']
 for name,h in fq['source_after'].items():assert rows[label+'/source/'+name]['sha256']==h
 assert fq['images_before']==fq['images_after']==q['images_after'] and fq['host_tools_before']==fq['host_tools_after']==q['host_tools_after']
 assert fq['producer_hosts_before']==fq['producer_hosts_after']==q['producer_hosts_after'] and fq['producer_before']==fq['producer_after']==q['producer_after']
 fr=namespace['replay'](lambda path,label=label:data(label+'/'+path.removeprefix('qualified/')),expected_pass=False)
 assert not fr['passed'] and fr['methods']==4 and fr['product_commands_executed']==0
# I execute only pure Python fixture arithmetic, never products, to independently
# audit the correction and preserve every original matrix byte as a prefix.
import ast,math,re,random,struct,yaml
schema=yaml.safe_load(data('qualified/source/spec/nanoisa.yaml'));ops={v['name']:v for v in schema['legacy_opcodes']}
def fixture_module(raw):
 tree=ast.parse(raw);items=[]
 for node in tree.body:
  if isinstance(node,ast.FunctionDef):items.append(node)
 ns=dict(OPS=ops,WIDTH={'U8':1,'U16':2,'U32':4,'I32':4,'I64':8,'F64':8},FMT={'U8':'B','U16':'H','U32':'I','I32':'i','I64':'q','F64':'Q'},struct=struct,random=random)
 exec(compile(ast.Module(body=items,type_ignores=[]),'fixture-functions-only','exec'),ns);return ns
before=fixture_module(data('failed-visit-fixture/source/tests/test_nanolang_translator_constant_reachability.py'))
after=fixture_module(data('qualified/source/tests/test_nanolang_translator_constant_reachability.py'))
old_cases,new_cases=before['cases'](),after['cases']();assert new_cases[:len(old_cases)]==old_cases and len(new_cases)>len(old_cases)
old_bounds,new_bounds=before['resource_cases'](),after['resource_cases']();assert len(old_bounds)==11 and len(new_bounds)==13
assert old_bounds[:9]==new_bounds[:9]
assert [v[1] for v in old_bounds[9:]]==[v[1] for v in new_bounds[9:11]]
assert [v[2] for v in old_bounds[9:]]==[1,0] and [v[2] for v in new_bounds[9:]]==[1,1,1,0]
for label,raw,want in new_bounds:
 count,imports,current,query=struct.unpack_from('<4I',raw);assert count==1 and imports==0 and current==0
 start,length,arity,locals_,results=struct.unpack_from('<5I',raw,16);code=raw[36:]
 assert start==arity==results==0 and length==len(code)
 if 'visits-' in label:
  actual=after['scan_visits'](code)
  assert actual in (261421,261422,262144,262145) and want==int(actual<=262144)
 if 'frame-constants-' in label:assert after['frame_charge'](code,locals_)==int(label.rsplit('-',1)[1])
 commands=[c for c in q['commands'] if c['argv'][-1].endswith('/'+label+'.bin')]
 assert len(commands)==7
 for c in commands:assert data('qualified/command-%03d.stdout'%c['number'])==(str(want)+'\n').encode()
# I retain original assembler inputs and resulting module/packet artifacts.
original=[c for c in q['commands'] if Path(c['argv'][0]).name=='nanoisa' and 'asm' in c['argv']]
assert len(original)==3 and all(c['exit']==0 for c in original)
assert b'extern fn' not in data('qualified/source/src_nano/nvm2c/constant_reachability.nano')
permutation=js('failed-permutation/qualification-result.json')
assert permutation['terminal'] and not permutation['accepted'] and permutation['all_guards_match']
assert (permutation['methods'],permutation['failures'],permutation['errors'])==(4,1,0)
assert not permutation['owned_cleanup']['remaining'] and not permutation['owned_cleanup']['records']
for name,h in q['source_after'].items():
 if name!='src_nano/nvm2c/constant_reachability.nano':assert permutation['source_after'][name]==h
assert failure['source_after']['src_nano/nvm2c/constant_reachability.nano']==permutation['source_after']['src_nano/nvm2c/constant_reachability.nano']
old_source=data('failed-permutation/source/src_nano/nvm2c/constant_reachability.nano').decode()
new_source=data('qualified/source/src_nano/nvm2c/constant_reachability.nano').decode()
restored=new_source[:new_source.index('\n    let dup_functions')]+'}\n'
assert restored.count('(== op 244)')==3
restored=restored.replace('(== op 244)','(== op 112)').replace('} else if (== op 7) { set output [ak,av,ak,av,0,0]','} else if (== op 8) { set output [ak,av,ak,av,0,0]')
assert restored==old_source
assert (ops['DUP']['code'],ops['POP']['code'],ops['PICK']['code'],ops['TUPLE_NEW']['code'])==(7,8,244,112)
# The original failing batch remains a literal C/native differential witness.
rc=data('failed-permutation/command-500.stdout').decode().splitlines()
nc=data('failed-permutation/command-502.stdout').decode().splitlines()
assert len(rc)==len(nc)==16 and [(i,a,b) for i,(a,b) in enumerate(zip(rc,nc)) if a!=b]==[(8,'1','0'),(11,'1','0')]
policy=js('qualified/shadow-policy.json');assert policy['exit']==0 and policy['staged_new_files'] and policy['script_sha256']==rows['collector-source/check_shadow_tests.sh']['sha256']
for name,h in policy['sources'].items():assert rows['qualified/source/'+name]['sha256']==h
assert js('qualified/make-dry-run.json')['exit']==0 and js('qualified/make-dry-run.json')['prerequisites_assumed_old']
assert data('qualified/make-dry-run.stdout')==b'python3 -m unittest -v tests.test_nanolang_translator_constant_reachability\n'
f=js('collector-source/refusal.json');assert f['actual_nonterminal_parent'] and not f['archive_created'] and f['exit']!=0
assert f['collector_sha256']==rows['collector-source/live-package.py']['sha256'] and f['parent_sha256']==rows['collector-source/live-parent.json']['sha256']
assert not js('collector-source/live-parent.json')['terminal'] and b'AssertionError' in data('collector-source/live-refusal.stderr')
print(json.dumps(dict(passed=True,records=len(rows),methods=4,commands=len(q['commands']),cases=len(new_cases),resource_cases=13,replay=r,component_only=True,whole_release=False)))
