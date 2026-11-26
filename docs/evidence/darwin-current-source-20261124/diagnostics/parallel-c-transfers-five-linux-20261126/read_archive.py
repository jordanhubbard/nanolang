"""I verify complete simultaneous scalar-transfer observations without executing products."""
from pathlib import Path
import hashlib,json,sys,types
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'370b8f618512471ffb7fb0f56015a560275fd637e77f5aff91878db95e95b67a')
pm,pr,pd,pj=read_archive(base.parent/'binary64-c-expressions-five-linux-20261126','882b1f54bb588c5dd82f7acd4912e2823eca3290d365f3722213ea1c62dc8360')
bm,br,bd,bj=read_archive(base.parent/'typechecker-vector-prefix-cold-bootstrap-linux-20261126','3c2f440e292cb2c0c54f8b3d344b239c927223bed505c874e90831b842f1c5d7')
assert m['component_only'] and m['pure_nanolang'] and m['full_five_routes']
assert not any(m[k] for k in ['new_bootstrap','profile_admission','body_emission','runtime_emission_complete','pure_cutover','paired','whole_release'])
assert m['source_commit']==pm['source_commit']==bm['source_commit']=='04cee11a861af22dfd4540f369eb2c6bb42a9cfd'
q=js('qualified/qualification-result.json');old=pj('qualified/qualification-result.json');s=js('qualified/preparation.json')
assert q['terminal'] and q['accepted'] and q['all_guards_match'] and q['tests_pass']
assert (q['methods'],q['failures'],q['errors'],q['skipped'])==(12,0,0,0)
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
make=data('qualified/source/Makefile.gnu');assert make.count(block)==1 and sha(make.replace(block,b'',1))==old['source_after']['Makefile.gnu']
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
   assert lines[i+1]=='    %s nano_aot_edge_%d = nano_aot_local_%d;'%('double' if tag==3 else 'int64_t',i,source)
   assert lines[count+i+1]=='    nano_aot_local_%d = nano_aot_edge_%d;'%(destination,i)
 packets+=1
assert (packets,accepted)==(523,497)
expected=data('qualified/command-023.stdout');assert len(expected.splitlines())==192384
assert all(data('qualified/command-%03d.stdout'%n)==expected for n in [23,25,28,30,33,35,38,40,43,45])
assert data('qualified/command-009.stdout')==data('qualified/command-012.stdout') and len(data('qualified/command-009.stdout').splitlines())==3006
clang=js('clang-qualified/result.json');assert clang['terminal'] and clang['accepted'] and clang['actual_retained_inputs'] and not clang['new_translation']
assert clang['expected_lines']==192384 and clang['compiler_before']==clang['compiler_after']==rows['external/'+clang['compiler'].lstrip('/')]['sha256']
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
assert f['collector_sha256']==rows['collector-source/package.py']['sha256'] and f['parent_sha256']==rows['collector-source/live-parent.json']['sha256']
assert not js('collector-source/live-parent.json')['terminal'] and b'AssertionError' in data('collector-source/live-refusal.stderr')
module=types.ModuleType('parallel_journal_replay');exec(compile(data('collector-source/replay.py'),'archived-replay.py','exec'),module.__dict__)
replay=module.replay(data);assert replay==js('qualified/replay.stdout')
print(json.dumps(dict(terminal=True,accepted=True,component_only=True,records=len(rows),methods=12,commands=162,replay=replay,transfer_reports=2615,valid_sets=497,gcc_results=1923840,clang_results=192384,reference_results=6012,whole_release=False,paired=False)))
