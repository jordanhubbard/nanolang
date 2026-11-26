"""I verify my full exact-graph observations without executing product commands."""
from pathlib import Path
import hashlib,json,sys,types
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'82f7fe2cdb6bb0c86db1ae3a5556b5b961b24aa2fd6e79097f92f0db02c0e9fb')
pm,pr,pd,pj=read_archive(base.parent/'parallel-c-transfers-canonical-five-linux-20261126','6dd2543455edeaaff2f857434129d2ae3940fb8fd6ea774b0e744ecec37e0f52')
q=js('qualified/qualification-result.json');old=pj('qualified/qualification-result.json');s=js('qualified/preparation.json')
assert m['component_only'] and m['pure_nanolang'] and m['full_five_routes']
assert not any(m[k] for k in ['new_bootstrap','profile_admission','body_emission','runtime_emission_complete','pure_cutover','paired','whole_release'])
assert m['source_commit']==pm['source_commit']=='04cee11a861af22dfd4540f369eb2c6bb42a9cfd'
assert q['terminal'] and q['accepted'] and q['all_guards_match'] and q['tests_pass']
assert (q['methods'],q['failures'],q['errors'],q['skipped'])==(18,0,0,0)
assert (q['ordinary_seconds'],q['build_seconds'],q['run_seconds'])==(10,600,60)
assert not q['other_jobs'] and not q['unexpected_sanitizer'] and q['free_bytes']>15*1024**3 and q['filesystem']!='nfs'
assert all(r['cpu']>=1 and r['share']>=.75 for r in q['cpu_rounds'])
assert not q['owned_cleanup']['remaining'] and not q['owned_cleanup']['records']
assert q['source_before']==q['source_after']==s['source_before'] and len(q['source_after'])==3985
new={'src_nano/nvm2c/shape_graph.nano','tests/nvm2c_shape_graph_driver.nano','tests/nvm2c_shape_graph_reference.c','tests/test_nanolang_translator_shape_graph.py'}
assert set(q['source_after'])==set(old['source_after'])|new
for name,h in q['source_after'].items():
 assert rows['qualified/source/'+name]['sha256']==h
 if name not in new|{'Makefile.gnu'}:assert old['source_after'][name]==h
block=b'.PHONY: test-nanolang-translator-shape-graph\ntest-nanolang-translator-shape-graph: $(COMPILER_C) nano_virt nano_vm nvm2c nvm2c-runtime\n\tpython3 -m unittest -v tests.test_nanolang_translator_shape_graph\n\ntest-units: test-nanolang-translator-shape-graph\n\n'
make=data('qualified/source/Makefile.gnu');assert make.count(block)==1 and make.replace(block,b'',1)==pd('qualified/source/Makefile.gnu')
assert q['images_before']==q['images_after']==old['images_after'] and len(q['images_after'])==12
for name,h in q['images_after'].items():assert rows['qualified/source/'+name]['sha256']==h
assert q['host_tools_before']==q['host_tools_after']==old['host_tools_after'] and len(q['host_tools_after'])==22
assert q['producer_before']==q['producer_after']==old['producer_after']=='f28b2bf49e94b5033e6b579b07f5663f0fc7151f1a933ab09efef53e0e13cd74'
assert q['producer_hosts_before']==q['producer_hosts_after']==old['producer_hosts_after'] and len(q['producer_hosts_after'])==6
assert q['producer_receipt_before']==q['producer_receipt_after']==old['producer_receipt_after']
for path,h in {**q['host_tools_after'],**q['producer_hosts_after'],q['producer']:q['producer_after']}.items():assert rows['external/'+path.lstrip('/')]['sha256']==h
assert len(q['commands'])==279
for n,c in enumerate(q['commands'],1):
 assert c['number']==n and c['streams_complete'] and c['exit']==(0 if n<275 else 2)
 assert c['effective_budget']==(600 if c['build'] else 60) and c['timeout_requested']==60
 assert not c['cleanup']['remaining'] and not c['cleanup']['records']
 for stream in ['stdout','stderr']:assert rows['qualified/command-%03d.%s'%(n,stream)]['sha256']==c[stream+'_sha256']
 for a in c['artifacts']:
  if a.get('kind')!='symlink':assert rows['qualified/retained/'+a['sha256']]['bytes']==a['bytes'] and rows['qualified/retained/'+a['sha256']]['sha256']==a['sha256']
# My journal replay reconstructs all packets, full expected reports and per-operation
# assertions by running the original permanent unittest logic with Popen forbidden.
namespace={};exec(compile(data('collector-source/replay.py'),'replay.py','exec'),namespace)
r=namespace['replay'](data);assert r['passed'] and r['methods']==18 and r['product_commands_executed']==0
# I additionally require complete 10000-deep facts, not a reduced-depth/checksum report.
for label in ['original-deep-10000-0.bin','original-deep-10000-1.bin','original-wide-300.bin']:
 commands=[c for c in q['commands'] if c['argv'][-1].endswith('/'+label)]
 assert len(commands)==7
 reports=[data('qualified/command-%03d.stdout'%c['number']) for c in commands]
 assert all(v==reports[0] for v in reports)
 lines=reports[0].splitlines();assert sum(v.startswith(b'Z|') for v in lines)==1
 want=0 if label=='original-deep-10000-1.bin' else (40006 if 'deep' in label else 1202)
 assert sum(v.startswith(b'N|') for v in lines)==want
# I retain each failed run and prove that the count-bound fix changed only the driver.
f=js('failed-count/qualification-result.json');assert f['terminal'] and not f['accepted'] and f['all_guards_match'] and f['failures']==1 and not f['errors'] and f['methods']==18
assert not f['owned_cleanup']['remaining'] and not f['owned_cleanup']['records']
for name,h in q['source_after'].items():
 if name!='tests/nvm2c_shape_graph_driver.nano':assert f['source_after'][name]==h
before=data('failed-count/source/tests/nvm2c_shape_graph_driver.nano').decode();after=data('qualified/source/tests/nvm2c_shape_graph_driver.nano').decode()
guard='    if (> count (/ (- (array_length bytes) 4) 8)) { return false }\n';shadow='    assert (== (framed [255,255,255,255]) false)\n'
assert after.count(guard)==after.count(shadow)==1 and after.replace(guard,'',1).replace(shadow,'',1)==before
fail=[c for c in f['commands'] if c.get('exit')==1 and c['argv'][-1].endswith('/bad-5.bin')];assert len(fail)==1
assert b'Out of memory' in data('failed-count/command-%03d.stderr'%fail[0]['number'])
for label in ['failed-driver','failed-python']:
 f=js(label+'/qualification-result.json');assert f['terminal'] and not f['accepted'] and f['all_guards_match']
 assert not f['owned_cleanup']['remaining'] and not f['owned_cleanup']['records']
assert not js('failed-python/qualification-result.json')['commands']
assert js('development-driver/result.json')['execute']['exit']==0
policy=js('qualified/shadow-policy.json');assert policy['exit']==0 and policy['staged_new_files']
assert policy['script_sha256']==rows['collector-source/check_shadow_tests.sh']['sha256']
for name,h in policy['sources'].items():assert rows['qualified/source/'+name]['sha256']==h
check=js('qualified/make-dry-run.json');assert check['exit']==0 and check['prerequisites_assumed_old']
assert data('qualified/make-dry-run.stdout')==b'python3 -m unittest -v tests.test_nanolang_translator_shape_graph\n'
i=js('interrupted-premature/interruption.json');assert i['terminal'] and not i['accepted'] and i['external_interruption'] and i['all_selected_inputs_match'] and not i['remaining'] and i['parent_absent'] and i['original_journal_not_rewritten']
assert not js('interrupted-premature/qualification-result.json')['accepted']
f=js('collector-source/refusal.json');assert f['actual_nonterminal_parent'] and not f['archive_created'] and f['exit']!=0
assert f['collector_sha256']==rows['collector-source/live-package.py']['sha256'] and f['parent_sha256']==rows['collector-source/live-parent.json']['sha256']
assert not js('collector-source/live-parent.json')['terminal'] and b'AssertionError' in data('collector-source/live-refusal.stderr')
print(json.dumps(dict(passed=True,records=len(rows),methods=18,commands=279,replay=r,component_only=True,whole_release=False)))
