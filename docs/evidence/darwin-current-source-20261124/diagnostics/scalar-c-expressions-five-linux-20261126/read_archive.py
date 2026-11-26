"""I verify my private scalar-expression receipt, including every original criterion."""
from pathlib import Path
import ast,hashlib,json,sys,types
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda raw:hashlib.sha256(raw).hexdigest()
m,rows,data,js=read_archive(base,'d409b1db854e3bd789ffec47fc19255b41694e0b88013689269f906e16fa056b')
pm,pr,pd,pj=read_archive(base.parent/'typechecker-vector-prefix-cold-bootstrap-linux-20261126','3c2f440e292cb2c0c54f8b3d344b239c927223bed505c874e90831b842f1c5d7')
assert m['component_only'] and m['pure_nanolang'] and m['full_five_routes'] and m['working_overlay']
assert not any(m[k] for k in ['new_bootstrap','profile_admission','body_emission','runtime_emission_complete','pure_cutover','paired','whole_release'])
assert m['source_commit']==pm['source_commit']=='04cee11a861af22dfd4540f369eb2c6bb42a9cfd'
q=js('qualified/qualification-result.json');prep=js('qualified/preparation.json')
assert q['terminal'] and q['accepted'] and q['all_guards_match'] and q['tests_pass']
assert (q['methods'],q['failures'],q['errors'],q['skipped'])==(5,0,0,0)
assert (q['ordinary_seconds'],q['build_seconds'],q['run_seconds'])==(10,600,60)
assert not q['unexpected_sanitizer'] and not q['other_jobs'] and q['free_bytes']>15*1024**3 and q['filesystem']!='nfs'
assert all(r['cpu']>=1 and r['share']>=.75 for r in q['cpu_rounds'])
assert not q['owned_cleanup']['remaining'] and not q['owned_cleanup']['records']
assert q['source_before']==q['source_after']==prep['source_before'] and len(q['source_before'])==3975
new={'src_nano/nvm2c/scalar_c.nano','tests/nvm2c_scalar_c_driver.nano','tests/test_nanolang_translator_scalar_c.py'}
old=pj('bootstrap/bootstrap-result.json')['source_after'];assert set(q['source_before'])==set(old)|new|{'spec/nanoisa.yaml'}
for name,h in q['source_after'].items():
 assert rows['qualified/source/'+name]['sha256']==h
 if name not in new|{'Makefile.gnu','spec/nanoisa.yaml'}:assert old[name]==h
block=b'.PHONY: test-nanolang-translator-scalar-c\ntest-nanolang-translator-scalar-c: $(COMPILER_C) nano_virt nano_vm nvm2c nvm2c-runtime\n\tpython3 -m unittest -v tests.test_nanolang_translator_scalar_c\n\ntest-units: test-nanolang-translator-scalar-c\n\n'
make=data('qualified/source/Makefile.gnu');assert make.count(block)==1 and sha(make.replace(block,b'',1))==old['Makefile.gnu']
assert q['images_before']==q['images_after'] and len(q['images_after'])==12
for name,h in q['images_after'].items():assert rows['qualified/source/'+name]['sha256']==pr['bootstrap/source/'+name]['sha256']==h
assert q['host_tools_before']==q['host_tools_after'] and len(q['host_tools_after'])==22
for path,h in {**q['host_tools_after'],**q['producer_hosts_after'],q['producer']:q['producer_after']}.items():assert rows['external/'+path.lstrip('/')]['sha256']==h
assert q['producer_before']==q['producer_after']=='f28b2bf49e94b5033e6b579b07f5663f0fc7151f1a933ab09efef53e0e13cd74'
boot=Path(pm['roots']['bootstrap'])
assert q['producer_receipt_before']==q['producer_receipt_after']==pr['bootstrap/'+str(Path(q['producer_receipt']).relative_to(boot))]['sha256']
assert q['producer_hosts_before']==q['producer_hosts_after'] and len(q['producer_hosts_after'])==6
receipt=pj('bootstrap/source/bin/nanoc_bootstrap.json')
for path,h in q['producer_hosts_after'].items():assert pr['bootstrap/'+str(Path(path).relative_to(boot))]['sha256']==h and path.encode() in pd('bootstrap/source/bin/nanoc_bootstrap.json')
assert len(q['commands'])==72
for index,c in enumerate(q['commands'],1):
 assert c['number']==index and c['streams_complete'] and c['exit']==(0 if index<68 else 2)
 assert c['effective_budget']==(600 if c['build'] else 60) and c['timeout_requested']==60
 assert not c['cleanup']['remaining'] and not c['cleanup']['records']
 for stream in ['stdout','stderr']:assert rows['qualified/command-%03d.%s'%(index,stream)]['sha256']==c[stream+'_sha256']
 for a in c['artifacts']:
  if a.get('kind')!='symlink':assert rows['qualified/retained/'+a['sha256']]['sha256']==a['sha256'] and rows['qualified/retained/'+a['sha256']]['bytes']==a['bytes']
reports=[data('qualified/command-%03d.stdout'%n) for n in [21,26,31,36,41]]
assert all(r==reports[0] for r in reports)
report=reports[0].split(b'PRELUDE-END\n')[1].splitlines();assert len(report)==415 and sum(r.split(b'|')[4]==b'1' for r in report)==71
assert all(data('qualified/command-%03d.stdout'%n)==reports[0] for n in [14,17,20])
assert data('qualified/command-009.stdout')==data('qualified/command-012.stdout') and len(data('qualified/command-009.stdout').splitlines())==1440
expected=data('qualified/command-023.stdout');assert len(expected.splitlines())==7174
assert all(data('qualified/command-%03d.stdout'%n)==expected for n in [23,25,28,30,33,35,38,40,43,45])
clang=js('clang-qualified/result.json');assert clang['terminal'] and clang['accepted'] and clang['expected_lines']==7174
assert clang['compiler_before']==clang['compiler_after']==rows['external/'+clang['compiler'].lstrip('/')]['sha256']
assert all(c['exit']==0 for c in clang['commands']) and all(f in clang['commands'][1]['argv'] for f in ['-std=c11','-Wall','-Wextra','-Werror','-O3'])
assert data('clang-qualified/execute.stdout')==data('clang-qualified/expected.stdout')==expected
assert rows['clang-qualified/native']['sha256']==clang['native_sha256']
for suffix,key in [('actual-emitted.c','source_sha256'),('actual-inputs.txt','input_sha256')]:
 assert rows['clang-qualified/'+suffix]['sha256']==clang[key]
 assert any(a['sha256']==clang[key] for a in q['commands'][21]['artifacts'])
assert sha(expected)==clang['expected_sha256']
for label,count,failures,commands in [('setup-refused',0,0,0),('oracle-refused',5,6,50),('alias-refused',5,5,57),('partial-functional',5,0,72)]:
 f=js(label+'/qualification-result.json');assert f['terminal'] and len(f['commands'])==commands
 if label=='setup-refused':assert 'methods' not in f and 'failures' not in f and f['exception']=="FileNotFoundError(2, 'No such file or directory')"
 else:assert f['methods']==count and f['failures']==failures
 assert not f['owned_cleanup']['remaining'] and not f['owned_cleanup']['records']
 if label!='partial-functional':assert not f['accepted']
 else:assert f['accepted'] and not f.get('producer_hosts_before')
for r in js('collector-source/refusals.json'):
 assert r['exit']!=0 and not r['archive_created'] and r['collector_sha256']==rows['collector-source/package.py']['sha256']
 assert rows['collector-source/'+r['label']+'-parent.json']['sha256']==r['parent_sha256']
 assert b'AssertionError' in data('collector-source/'+r['label']+'-refusal.stderr')
 assert data('collector-source/'+r['label']+'-parent.json')==data(('partial-functional' if r['label']=='missing-host-guards' else r['label']+'-refused')+'/qualification-result.json')
h=js('qualified/harness-correction.json');text=data('qualified/source/tests/test_nanolang_translator_scalar_c.py')
assert sha(text)==h['new_sha256']
for before,after in reversed(h['changed_regions']):assert text.count(after.encode())==1;text=text.replace(after.encode(),before.encode(),1)
assert text==data('oracle-refused/source/tests/test_nanolang_translator_scalar_c.py') and sha(text)==h['old_sha256']
def functions(raw):return {n.name:ast.dump(n) for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
a=functions(text);b=functions(data('qualified/source/tests/test_nanolang_translator_scalar_c.py'))
for name in h['unchanged_functions']:assert a[name]==b[name]
e=js('qualified/emitter-alias-repair.json');text=data('qualified/source/src_nano/nvm2c/scalar_c.nano');assert sha(text)==e['new_sha256']
for k in ['shadow_insertion','production_insertion']:assert text.count(e[k].encode())==1;text=text.replace(e[k].encode(),b'',1)
assert text==data('alias-refused/source/src_nano/nvm2c/scalar_c.nano') and sha(text)==e['old_sha256']
assert data('alias-refused/source/tests/test_nanolang_translator_scalar_c.py')==data('qualified/source/tests/test_nanolang_translator_scalar_c.py')
policy=js('qualified/shadow-policy.json');assert policy['exit']==0 and policy['staged_new_files'] and policy['script_sha256']==rows['collector-source/check_shadow_tests.sh']['sha256']
for name,h in policy['source_sha256'].items():assert rows['qualified/source/'+name]['sha256']==h
assert data('partial-functional/make-recipe.stdout')==b'python3 -m unittest -v tests.test_nanolang_translator_scalar_c\n'
assert data('partial-functional/source/Makefile.gnu')==make
module=types.ModuleType('scalar_journal_replay');exec(compile(data('collector-source/replay.py'),'archived-replay.py','exec'),module.__dict__)
replayed=module.replay(data);assert replayed==js('qualified/replay.stdout')
print(json.dumps(dict(terminal=True,accepted=True,component_only=True,records=len(rows),methods=5,commands=72,replay=replayed,scalar_reports=2075,valid_expressions=71,gcc_results=71740,clang_results=7174,reference_results=2880,paired=False,whole_release=False)))
