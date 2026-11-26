"""I verify complete simultaneous scalar-transfer observations without executing products."""
from pathlib import Path
import hashlib,json,sys,types
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'6dd2543455edeaaff2f857434129d2ae3940fb8fd6ea774b0e744ecec37e0f52')
pm,pr,pd,pj=read_archive(base.parent/'parallel-c-transfers-five-linux-20261126','370b8f618512471ffb7fb0f56015a560275fd637e77f5aff91878db95e95b67a')
bm,br,bd,bj=read_archive(base.parent/'typechecker-vector-prefix-cold-bootstrap-linux-20261126','3c2f440e292cb2c0c54f8b3d344b239c927223bed505c874e90831b842f1c5d7')
assert m['component_only'] and m['pure_nanolang'] and m['full_five_routes']
assert not any(m[k] for k in ['new_bootstrap','profile_admission','body_emission','runtime_emission_complete','pure_cutover','paired','whole_release'])
assert m['source_commit']==pm['source_commit']==bm['source_commit']=='04cee11a861af22dfd4540f369eb2c6bb42a9cfd'
q=js('qualified/qualification-result.json');old=pj('qualified/qualification-result.json');s=js('qualified/preparation.json')
assert q['terminal'] and q['accepted'] and q['all_guards_match'] and q['tests_pass']
assert (q['methods'],q['failures'],q['errors'],q['skipped'])==(13,0,0,0)
assert (q['ordinary_seconds'],q['build_seconds'],q['run_seconds'])==(10,600,60)
assert not q['other_jobs'] and not q['unexpected_sanitizer'] and q['free_bytes']>15*1024**3 and q['filesystem']!='nfs'
assert all(r['cpu']>=1 and r['share']>=.75 for r in q['cpu_rounds'])
assert not q['owned_cleanup']['remaining'] and not q['owned_cleanup']['records']
assert q['source_before']==q['source_after']==s['source_before'] and len(q['source_after'])==3981
new={'src_nano/nvm2c/parallel_c.nano','tests/nvm2c_parallel_c_driver.nano','tests/test_nanolang_translator_parallel_c.py'}
assert set(q['source_after'])==set(old['source_after'])|new
for name,h in q['source_after'].items():
 assert rows['qualified/source/'+name]['sha256']==h
 if name not in new|{'Makefile.gnu'}:assert old['source_after'][name]==h
block=b'.PHONY: test-nanolang-translator-parallel-c\ntest-nanolang-translator-parallel-c: $(COMPILER_C) nano_virt nano_vm nvm2c nvm2c-runtime\n\tpython3 -m unittest -v tests.test_nanolang_translator_parallel_c\n\ntest-units: test-nanolang-translator-parallel-c\n\n'
make=data('qualified/source/Makefile.gnu');assert make.count(block)==1 and sha(make)==old['source_after']['Makefile.gnu']
assert q['images_before']==q['images_after']==old['images_after'] and len(q['images_after'])==12
for name,h in q['images_after'].items():assert rows['qualified/source/'+name]['sha256']==h
assert q['host_tools_before']==q['host_tools_after']==old['host_tools_after'] and len(q['host_tools_after'])==22
assert q['producer_before']==q['producer_after']==old['producer_after']=='f28b2bf49e94b5033e6b579b07f5663f0fc7151f1a933ab09efef53e0e13cd74'
assert q['producer_hosts_before']==q['producer_hosts_after']==old['producer_hosts_after'] and len(q['producer_hosts_after'])==6
assert q['producer_receipt_before']==q['producer_receipt_after']==old['producer_receipt_after']==br['bootstrap/source/bin/nanoc_bootstrap.json']['sha256']
for path,h in {**q['host_tools_after'],**q['producer_hosts_after'],q['producer']:q['producer_after']}.items():assert rows['external/'+path.lstrip('/')]['sha256']==h
assert len(q['commands'])==162
for n,c in enumerate(q['commands'],1):
 assert c['number']==n and c['streams_complete'] and c['exit']==(0 if n<158 else 2)
 assert c['effective_budget']==(600 if c['build'] else 60) and c['timeout_requested']==60
 assert not c['cleanup']['remaining'] and not c['cleanup']['records']
 for stream in ['stdout','stderr']:assert rows['qualified/command-%03d.%s'%(n,stream)]['sha256']==c[stream+'_sha256']
 for a in c['artifacts']:
  if a.get('kind')!='symlink':assert rows['qualified/retained/'+a['sha256']]['bytes']==a['bytes'] and rows['qualified/retained/'+a['sha256']]['sha256']==a['sha256']
reports=[data('qualified/command-%03d.stdout'%n) for n in [21,26,31,36,41]];assert all(r==reports[0] for r in reports)
assert all(data('qualified/command-%03d.stdout'%n)==reports[0] for n in [14,17,20])
import io
stream=io.BytesIO(reports[0]);assert stream.readline()==b'PRELUDE-BEGIN\n' and stream.readline()==b'#include <stdint.h>\n' and stream.readline()==b'PRELUDE-END\n'
packets=accepted=0
while stream.tell()<len(reports[0]):
 header=stream.readline().rstrip(b'\n').split(b'|');assert len(header)==8 and header[0]==b'T'
 size=int(header[7]);text=stream.read(size);assert len(text)==size and stream.readline()==b'END\n'
 columns=[[int(x) for x in part.split(b',')] if part else [] for part in header[2:5]]
 if header[5]==b'0':assert header[6]==b'0' and not text
 else:
  assert header[5]==b'1';accepted+=1;destinations,sources,tags=columns;count=int(header[6]);assert len(destinations)==len(sources)==len(tags)==count
  lines=text.decode().splitlines();assert len(lines)==2*count+2 and lines[0]=='{' and lines[-1]=='}'
  for i,(destination,source,tag) in enumerate(zip(destinations,sources,tags)):
   assert tag in (1,3,4)
   assert lines[i+1]=='    %s nano_aot_edge_%d = nano_aot_local_%d;'%('double' if tag==3 else 'int64_t',i,source)
   assert lines[count+i+1]=='    nano_aot_local_%d = nano_aot_edge_%d;'%(destination,i)
 packets+=1
assert (packets,accepted)==(947,501)
expected=data('qualified/command-023.stdout');assert len(expected.splitlines())==192896
assert all(data('qualified/command-%03d.stdout'%n)==expected for n in [23,25,28,30,33,35,38,40,43,45])
assert data('qualified/command-009.stdout')==data('qualified/command-012.stdout') and len(data('qualified/command-009.stdout').splitlines())==3014
clang=js('clang-qualified/result.json');assert clang['terminal'] and clang['accepted'] and clang['actual_retained_inputs'] and not clang['new_translation']
assert clang['expected_lines']==192896 and clang['compiler_before']==clang['compiler_after']==rows['external/'+clang['compiler'].lstrip('/')]['sha256']
assert data('clang-qualified/execute.stdout')==data('clang-qualified/expected.stdout')==expected
assert rows['clang-qualified/native']['sha256']==clang['native_sha256']
for file,key in [('actual-emitted.c','source_sha256'),('actual-inputs.txt','input_sha256')]:
 assert rows['clang-qualified/'+file]['sha256']==clang[key] and any(a['sha256']==clang[key] for a in q['commands'][21]['artifacts'])
assert sha(expected)==clang['expected_sha256'] and not clang['cleanup']['remaining'] and not clang['cleanup']['records']
assert [c['label'] for c in clang['commands']]==['version','loader-dependencies','header-dependencies','strict-optimized-build','execute']
for c in clang['commands']:assert c['exit']==0
assert all(f in clang['commands'][3]['argv'] for f in ['-std=c11','-Wall','-Wextra','-Werror','-O3','-MD','-MF'])
for category in ['tools','headers','host_tools']:
 assert clang[category+'_before']==clang[category+'_after']
 for path,h in clang[category+'_after'].items():
  if category=='host_tools':assert q['host_tools_after'][path]==h
  else:assert rows['clang-qualified/'+category+'/'+h]['sha256']==h
policy=js('qualified/shadow-policy.json');assert policy['exit']==0 and policy['staged_new_files'] and policy['script_sha256']==rows['collector-source/check_shadow_tests.sh']['sha256']
for name,h in policy['sources'].items():assert rows['qualified/source/'+name]['sha256']==h
makecheck=js('qualified/make-dry-run.json');assert makecheck['exit']==0 and makecheck['prerequisites_assumed_old'] and makecheck['make_sha256']==q['host_tools_after']['/usr/bin/make']
assert data('qualified/make-dry-run.stdout')==b'python3 -m unittest -v tests.test_nanolang_translator_parallel_c\n'
f=js('collector-source/refusal.json');assert f['actual_nonterminal_parent'] and not f['archive_created'] and f['exit']!=0
assert f['collector_sha256']==rows['collector-source/live-package.py']['sha256'] and f['parent_sha256']==rows['collector-source/live-parent.json']['sha256']
assert not js('collector-source/live-parent.json')['terminal'] and b'AssertionError' in data('collector-source/live-refusal.stderr')
correction=js('collector-source/collector-correction.json');before=data('collector-source/live-package.py').decode();after=data('collector-source/package.py').decode()
assert correction['changed_regions']==1 and correction['product_commands_reexecuted']==0 and before.count(correction['old'])==1 and before.replace(correction['old'],correction['new'],1)==after
failure_raw=(base/'collector-failure.json').read_bytes();assert sha(failure_raw)=='a2ef74525478870c9fb9bacafcfdf0ae0b4c8f0fcb4660d12c556e7c1aa27a8c';failure=json.loads(failure_raw)
assert failure['failed_script_sha256']==rows['collector-source/live-package.py']['sha256'] and not failure['partial_manifest_created'] and failure['product_commands_reexecuted']==0
for name in ['stdout','stderr']:assert sha((base/('collector-failure.'+name)).read_bytes())==failure[name+'_sha256']
assert b"KeyError: 'compiler_after'" in (base/'collector-failure.stderr').read_bytes()
assert failure['partial_objects_identical_to_final']
for item in failure['partial_objects_identical_to_final']:
 raw=(base/item['path']).read_bytes();assert len(raw)==item['bytes'] and sha(raw)==item['sha256']


# I prove the canonical identity independently of the earlier self-consistent mapping.
import ast,itertools,yaml
schema=yaml.safe_load(data('qualified/source/spec/nanoisa.yaml'));tags={v['name']:v['code'] for v in schema['value_tags']}
assert (tags['i64'],tags['u8'],tags['f64'],tags['bool'])==(1,2,3,4) and len(tags)==16
bug=js('tag-audit/result.json');assert bug['terminal'] and bug['accepted_diagnostic'] and bug['canonical_tag_bug_observed'] and bug['product_commands_executed']==0 and bug['canonical_tags']==tags
assert bug['parent_manifest']=='370b8f618512471ffb7fb0f56015a560275fd637e77f5aff91878db95e95b67a' and bug['schema_sha256']==sha(data('qualified/source/spec/nanoisa.yaml'))==sha(data('tag-audit/canonical-schema.yaml'))
for item in bug['reports']:
 assert item['sha256']==rows['tag-audit/%03d.stdout'%item['command']]['sha256']==pr['qualified/command-%03d.stdout'%item['command']]['sha256']
 assert item['u8_wrongly_accepted']['valid']==1 and item['boolean_wrongly_refused']['valid']==0
old_source=data('tag-audit/before/src_nano/nvm2c/parallel_c.nano').decode();new_source=data('qualified/source/src_nano/nvm2c/parallel_c.nano').decode()
assert sha(old_source.encode())==old['source_after']['src_nano/nvm2c/parallel_c.nano']
def nano_function(source,name):
 start=source.index('pub fn '+name+'(') if 'pub fn '+name+'(' in source else source.index('fn '+name+'(')
 return source[start:source.index('\nshadow '+name+' {',start)]
repair=js('tag-audit/repair-scope.json');assert repair['emit_algorithm_unchanged']
assert nano_function(old_source,'emit')==nano_function(new_source,'emit') and sha(nano_function(new_source,'emit').encode())==repair['emit_sha256']
old_guard,new_guard=repair['production_replacement'];assert nano_function(old_source,'c_type').count(old_guard)==1
assert nano_function(old_source,'c_type').replace(old_guard,new_guard,1)==nano_function(new_source,'c_type')
for name in ['refused','scratch','columns_valid','prelude']:assert nano_function(old_source,name)==nano_function(new_source,name)
old_py=data('tag-audit/before/tests/test_nanolang_translator_parallel_c.py');new_py=data('qualified/source/tests/test_nanolang_translator_parallel_c.py')
assert sha(old_py)==old['source_after']['tests/test_nanolang_translator_parallel_c.py']
def functions(raw):return {n.name:n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
old_functions,new_functions=functions(old_py),functions(new_py)
for name in ['expected','reports','execution_source']:assert ast.dump(old_functions[name])==ast.dump(new_functions[name])
def corpus_from(nodes):
 namespace=dict(itertools=itertools,INTEGER=1,U8=2,FLOAT=3,BOOLEAN=4)
 exec(compile(ast.Module(body=[nodes['corpus']],type_ignores=[]),'corpus-only','exec'),namespace)
 return namespace['corpus']()
old_cases,new_cases=corpus_from(old_functions),corpus_from(new_functions);assert len(old_cases)==523 and len(new_cases)==947 and new_cases[:523]==old_cases
assert new_cases[-258:]==[(f'wire-tag-{tag}',[0],[1],[tag]) for tag in range(-1,257)]
def packet_rows(raw):
 f=io.BytesIO(raw)
 while f.readline()!=b'PRELUDE-END\n':pass
 result=[]
 while f.tell()<len(raw):
  h=f.readline().rstrip(b'\n').split(b'|');text=f.read(int(h[7]));assert f.readline()==b'END\n';result.append((h,text))
 return result
prior_rows,current_rows=packet_rows(pd('qualified/command-021.stdout')),packet_rows(reports[0])
assert len(prior_rows)==523 and all(a[0][:5]==b[0][:5] for a,b in zip(prior_rows,current_rows[:523]))
changes=sum(a[0][5]!=b[0][5] for a,b in zip(prior_rows,current_rows[:523]));assert changes==167
for tag,(h,text) in zip(range(-1,257),current_rows[-258:]):assert h[5]==(b'1' if tag in (1,3,4) else b'0')
assert js('qualified/shadow-policy.json')['isolated_added_source_check'] and sum(map(len,js('qualified/shadow-policy.json')['functions_checked'].values()))==14
module=types.ModuleType('parallel_journal_replay');exec(compile(data('collector-source/replay.py'),'archived-replay.py','exec'),module.__dict__)
replay=module.replay(data);assert replay==js('qualified/replay.stdout')
print(json.dumps(dict(terminal=True,accepted=True,component_only=True,records=len(rows),methods=13,commands=162,replay=replay,transfer_reports=4735,valid_sets=501,gcc_results=1928960,clang_results=192896,reference_results=6028,whole_release=False,paired=False)))
