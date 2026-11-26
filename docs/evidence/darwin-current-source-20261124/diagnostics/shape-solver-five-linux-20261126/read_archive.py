"""I verify my full exact-graph observations without executing product commands."""
from pathlib import Path
import hashlib,json,sys,types
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'aed0f2767d13142d8d58fcc30af21d69a7b948a8caf13b320d1d09e725bc5874')
pm,pr,pd,pj=read_archive(base.parent/'shape-core-five-linux-20261126','82f7fe2cdb6bb0c86db1ae3a5556b5b961b24aa2fd6e79097f92f0db02c0e9fb')
q=js('qualified/qualification-result.json');old=pj('qualified/qualification-result.json');s=js('qualified/preparation.json')
assert m['component_only'] and m['pure_nanolang'] and m['full_five_routes']
assert not any(m[k] for k in ['new_bootstrap','profile_admission','body_emission','runtime_emission_complete','pure_cutover','paired','whole_release'])
assert m['source_commit']==pm['source_commit']=='04cee11a861af22dfd4540f369eb2c6bb42a9cfd'
assert q['terminal'] and q['accepted'] and q['all_guards_match'] and q['tests_pass']
assert (q['methods'],q['failures'],q['errors'],q['skipped'])==(22,0,0,0)
assert (q['ordinary_seconds'],q['build_seconds'],q['run_seconds'])==(10,600,60)
assert not q['other_jobs'] and not q['unexpected_sanitizer'] and q['free_bytes']>15*1024**3 and q['filesystem']!='nfs'
assert all(r['cpu']>=1 and r['share']>=.75 for r in q['cpu_rounds'])
assert not q['owned_cleanup']['remaining'] and not q['owned_cleanup']['records']
assert q['source_before']==q['source_after']==s['source_before'] and len(q['source_after'])==3989
new={'tests/nvm2c_shape_solver_driver.nano','tests/nvm2c_shape_solver_reference.c','tests/nvm2c_shape_solver_original_trace.c','tests/test_nanolang_translator_shape_solver.py'}
assert set(q['source_after'])==set(old['source_after'])|new
for name,h in q['source_after'].items():
 assert rows['qualified/source/'+name]['sha256']==h
 if name not in new|{'Makefile.gnu','src_nano/nvm2c/shape_graph.nano'}:assert old['source_after'][name]==h
block=b'.PHONY: test-nanolang-translator-shape-solver\ntest-nanolang-translator-shape-solver: $(COMPILER_C) nano_virt nano_vm nvm2c nvm2c-runtime\n\tpython3 -m unittest -v tests.test_nanolang_translator_shape_solver\n\ntest-units: test-nanolang-translator-shape-solver\n\n'
make=data('qualified/source/Makefile.gnu');assert make.count(block)==1 and make.replace(block,b'',1)==pd('qualified/source/Makefile.gnu')
assert q['images_before']==q['images_after']==old['images_after'] and len(q['images_after'])==12
for name,h in q['images_after'].items():assert rows['qualified/source/'+name]['sha256']==h
assert q['host_tools_before']==q['host_tools_after']==old['host_tools_after'] and len(q['host_tools_after'])==22
assert q['producer_before']==q['producer_after']==old['producer_after']=='f28b2bf49e94b5033e6b579b07f5663f0fc7151f1a933ab09efef53e0e13cd74'
assert q['producer_hosts_before']==q['producer_hosts_after']==old['producer_hosts_after'] and len(q['producer_hosts_after'])==6
assert q['producer_receipt_before']==q['producer_receipt_after']==old['producer_receipt_after']
for path,h in {**q['host_tools_after'],**q['producer_hosts_after'],q['producer']:q['producer_after']}.items():assert rows['external/'+path.lstrip('/')]['sha256']==h
assert len(q['commands'])==384
for n,c in enumerate(q['commands'],1):
 assert c['number']==n and c['streams_complete'] and c['exit']==(0 if n<380 else 2)
 assert c['effective_budget']==(600 if c['build'] else 60) and c['timeout_requested']==60
 assert not c['cleanup']['remaining'] and not c['cleanup']['records']
 for stream in ['stdout','stderr']:assert rows['qualified/command-%03d.%s'%(n,stream)]['sha256']==c[stream+'_sha256']
 for a in c['artifacts']:
  if a.get('kind')!='symlink':assert rows['qualified/retained/'+a['sha256']]['bytes']==a['bytes'] and rows['qualified/retained/'+a['sha256']]['sha256']==a['sha256']
# I retain every old exact-core operation byte for byte, except adding new error cases.
source=data('qualified/source/src_nano/nvm2c/shape_graph.nano').decode()
prefix=source[:source.index('\n# Solver header:')]
a=prefix.index('    if (== code 11)');b=prefix.index('    if (== code 6)',a)
prefix=prefix[:a]+prefix[b:];prefix=prefix.replace('then a lazy solver-header link.','then a reserved solver flag.')
assert prefix==pd('qualified/source/src_nano/nvm2c/shape_graph.nano').decode()
assert 'extern fn' not in source
# I reconstruct every permanent test and input, forbidding real subprocesses.
namespace={};exec(compile(data('collector-source/replay.py'),'replay.py','exec'),namespace)
r=namespace['replay'](data);assert r['passed'] and r['methods']==22 and r['product_commands_executed']==0
pre=js('pre-diagnostic/qualification-result.json');assert pre['terminal'] and pre['accepted'] and pre['all_guards_match'] and pre['methods']==22
assert not pre['owned_cleanup']['remaining'] and not pre['owned_cleanup']['records']
changed={'tests/nvm2c_shape_solver_driver.nano','tests/nvm2c_shape_solver_reference.c','tests/test_nanolang_translator_shape_solver.py'}
for name,h in q['source_after'].items():
 if name not in changed:assert pre['source_after'][name]==h
assert data('pre-diagnostic/source/tests/nvm2c_shape_solver_driver.nano').replace(b'} else if (== opcode 21) { set lines (array_push lines (solver_snapshot g))',b'} else if (== opcode 21) {\n                set lines (array_push lines (event "P" item index 0 output g))\n                set lines (array_push lines (solver_snapshot g))')==data('qualified/source/tests/nvm2c_shape_solver_driver.nano')
assert data('pre-diagnostic/source/tests/nvm2c_shape_solver_reference.c').replace(b'case 21:solver_snapshot(&g);break;',b'case 21:event("P",item,i,0,output,&g);solver_snapshot(&g);break;')==data('qualified/source/tests/nvm2c_shape_solver_reference.c')
import struct,collections
for index in (0,1):
 suffix='/original-%d.bin'%index
 commands=[c for c in q['commands'] if c['argv'][-1].endswith(suffix)]
 prior=[c for c in pre['commands'] if c['argv'][-1].endswith(suffix)]
 assert len(commands)==len(prior)==8
 assert data('qualified/command-%03d.stdout'%commands[0]['number'])==b'Shape constraints: 3890 passed, 0 failed\n'
 def retained_packet(label,command):
  artifacts=[a for a in command['artifacts'] if a['path'].endswith(suffix)]
  assert len(artifacts)==1
  return data(label+'/retained/'+artifacts[0]['sha256'])
 raw=retained_packet('qualified',commands[0]);assert raw==retained_packet('pre-diagnostic',prior[0])
 count=struct.unpack_from('<I',raw)[0];assert count==232
 cursor=4;h=collections.Counter();operations=0
 for _ in range(count):
  mode,size=struct.unpack_from('<II',raw,cursor);cursor+=8;assert mode==1
  for _ in range(size):
   op,a,b,c,want,check=struct.unpack_from('<6I',raw,cursor);cursor+=24;assert check==1;h[op]+=1;operations+=1
 assert cursor==len(raw)==3190020 and operations==132840 and set(range(14,28))<=h.keys()
 reports=[data('qualified/command-%03d.stdout'%c['number']) for c in commands[1:]];assert len(reports)==7 and all(v==reports[0] for v in reports)
 report=reports[0];lines=report.splitlines();assert sum(v.startswith(b'Z|') for v in lines)==232
 assert sum(v.startswith(b'P|') for v in lines)==h[21]==791
 assert b'I found conflicting aggregate shape kinds optional/string at nodes 1/2' in report
 assert b'N|40006|' in report and b'W|' in report and b'R|' in report
# I check the original 10000/300 core reports too, with unchanged complete bounds.
for label in ['original-deep-10000-0.bin','original-deep-10000-1.bin','original-wide-300.bin']:
 commands=[c for c in q['commands'] if c['argv'][-1].endswith('/'+label)];assert len(commands)==7
 reports=[data('qualified/command-%03d.stdout'%c['number']) for c in commands];assert all(v==reports[0] for v in reports)
 want=0 if label=='original-deep-10000-1.bin' else (40006 if 'deep' in label else 1202)
 assert sum(v.startswith(b'N|') for v in reports[0].splitlines())==want
policy=js('qualified/shadow-policy.json');assert policy['exit']==0 and policy['staged_new_files']
assert policy['script_sha256']==rows['collector-source/check_shadow_tests.sh']['sha256']
for name,h in policy['sources'].items():assert rows['qualified/source/'+name]['sha256']==h
assert js('qualified/make-dry-run.json')['exit']==0 and js('qualified/make-dry-run.json')['prerequisites_assumed_old']
assert data('qualified/make-dry-run.stdout')==b'python3 -m unittest -v tests.test_nanolang_translator_shape_solver\n'
for label,want in [('development-reserved',1),('development-shadows',0)]:assert js(label+'/result.json')['build']['exit']==want
assert not js('development-warning/result.json')['accepted'] and js('development-comparison/result.json')['accepted']
assert b'Expected parameter name' in data('development-reserved/build.stderr')
assert b'misleading-indentation' in data('development-warning/trace-build.stderr')
f=js('collector-source/refusal.json');assert f['actual_nonterminal_parent'] and not f['archive_created'] and f['exit']!=0
assert f['collector_sha256']==rows['collector-source/live-package.py']['sha256'] and f['parent_sha256']==rows['collector-source/live-parent.json']['sha256']
assert not js('collector-source/live-parent.json')['terminal'] and b'AssertionError' in data('collector-source/live-refusal.stderr')
print(json.dumps(dict(passed=True,records=len(rows),methods=22,commands=384,replay=r,original_assertions=3890,original_programs=232,original_operations=132840,component_only=True,whole_release=False)))
