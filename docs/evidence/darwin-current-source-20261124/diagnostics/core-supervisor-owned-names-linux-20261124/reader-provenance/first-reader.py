"""I verify retained failures and exact-source component results without claiming release acceptance."""
from pathlib import Path
import sys,json,subprocess,hashlib,re
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'23f16688cb4493c92a62b62f709b34d0d992ca09bf4f8611f653d05478e4d0f9');sha=lambda b:hashlib.sha256(b).hexdigest()
q=js('ownership-cleanup/cold-matrix-result.json')
assert q['terminal'] and q['all_three_profiles_executed'] and not q['component_accepted']
assert not m['complete_san_acceptance'] and not m['whole_release'] and not m['required_stage2_complete'] and not m['bootstrap_started'] and not m['full_source_retry'] and m['known_open_failure']==1252
assert q['candidate']==m['candidate'] and q['pin']==m['source_base']
assert q['tools']==q['tools_after']
for n,h in q['tools'].items():assert rows['tool/'+n.lstrip('/')]['sha256']==h
assert [p['mode'] for p in q['profiles']]==['normal','san','production']
assert [p['focused']['exit'] for p in q['profiles']]==[0,2,0]
for p in q['profiles']:
 mode=p['mode'];prefix='ownership-cleanup/cold-source-'+mode+'/'
 assert p['build']['exit']==0 and not p['build']['timed_out'] and not p['focused']['timed_out'] and not p['provider_copies'] and not p['original_providers_after']
 assert p['before']==p['after']
 for n,h in p['before'].items():assert rows[prefix+n]['sha256']==h
 for n,h in q['candidate'].items():assert p['before'][n]==h
 for name,h in p['producer_images'].items():
  assert rows[prefix+'bin/'+name]['sha256']==h
  head=data(prefix+'bin/'+name)[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
 for n,v in p['outputs'].items():assert rows[prefix+n]['sha256']==v['sha256'] and rows[prefix+n]['bytes']==v['bytes']
 stderr=data('ownership-cleanup/cold-'+mode+'-focused.stderr')
 for count in [13,10,6]:assert re.search(rb'Ran '+str(count).encode()+rb' tests in [^\n]+\n\nOK\n',stderr)
 assert re.search(rb'Ran 42 tests in [^\n]+\n\n'+(rb'FAILED \(failures=1\)' if mode=='san' else rb'OK\n'),stderr)
 assert b'LeakSanitizer' not in stderr and b'heap-use-after-free' not in stderr
 if mode=='san':
  assert b"damage='directory'" in stderr and b'allocation-size-too-big' in stderr and b'load_hash_cache' in stderr and b'0x8000000000000000' in stderr
 else:assert b'ERROR: AddressSanitizer' not in stderr
 cases=[n for n in rows if n.startswith(prefix+'.test_output/shadow-reap-') and n.endswith('/process.json')];assert len(cases)==3
 assert {n.rsplit('/',2)[1] for n in cases}=={'cseed','nanovirt','runtime'}
 for n in cases:
  process=js(n);obs=process['observations'];assert process['returncode']==1 and process['deadline_seconds']==1 and obs['child_finished_monotonic']-obs['parent_stopped_monotonic']>1
  folder=n.rsplit('/',1)[0];error=data(folder+'/stderr');assert b'after 1 seconds' in error
  if '/runtime/' not in n:assert data(folder+'/product')==b'I retain the prior output after late completion.\n'
  if '/cseed/' in n:
   events=[json.loads(l) for l in error.splitlines() if l.startswith(b'{"event":"cseed-shadow-phase"')];s=events[-1]
   assert s['parent_timeout'] and s['completed'] and s['reaped'] and s['exit_code']==s['signal']==0 and not s['clock_failure']
a=js('ownership-cleanup/sanitizer-classification.json');assert len(a['failures'])==1 and not a['leak_sanitizer'] and not a['use_after_free'] and a['allocation_size_too_big']
for label,count in [('extern-leaks',43),('extern-only-cleanup',18)]:
 old=js(label+'/cold-matrix-result.json');assert old['terminal'] and old['profiles'][-1]['mode']=='san' and old['profiles'][-1]['focused']['exit']!=0
 err=data(label+'/cold-san-focused.stderr');assert b'LeakSanitizer' in err and ('FAILED (failures=%d)'%count).encode() in err
red=js('third-red/red-result.json');assert red['exit']==1 and red['expected_failures']==1
assert b'Ran 3 tests' in data('third-red/red.stderr') and b'FAILED (failures=1)' in data('third-red/red.stderr')
old=js('nanovirt-counterexample/result.json');prod=js('nanovirt-counterexample/product-result.json')
assert old['terminal'] and old['before']==old['after'] and old['exit']==0 and old['parent_wchan_before_stop']=='hrtimer_nanosleep'
assert old['child_exit_observed_monotonic']-old['parent_stopped_monotonic']>1 and prod['child_exit_after_parent_stopped_seconds']>1
assert all(c['exit']==0 for c in prod['commands']) and rows['nanovirt-counterexample/product']['sha256']==prod['sha256']
setup=js('prepared386/setup-result.json');prep=js('prepared386/prepare-result.json');assert setup['admitted_setup'] and prep['accepted'] and prep['terminal']
assert setup['bootstrap_stage_seconds']==1800 and setup['bootstrap_shadow_seconds']==30 and setup['ordinary_shadow_seconds']==10
assert not any(n.startswith('prepared386/') and any(x in Path(n).name for x in ['bootstrap-receipt','nanoc_stage2','nanoc_stage3']) for n in rows)
w=js('hash-directory-witness/result.json');v=w['observation'];assert not v['regular'] and v['seek_end']==v['seek_start']==0 and v['ftell']==9223372036854775807 and v['allocation_size_hex']=='0x8000000000000000'
assert rows['hash-directory-witness/probe']['sha256']==w['probe_sha256'] and all(c['exit']==0 for c in w['commands'])
assert all(p['before']['src/module_builder.c']==w['module_builder_sha256'] for p in q['profiles'])
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(spec):
 proc.stdin.write((spec+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',spec;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';return raw
assert set(subprocess.check_output(['git','diff','--name-only',m['source_base'],m['source_commit']],cwd=repo,text=True).splitlines())==set(q['candidate'])
for n,h in q['profiles'][0]['before'].items():assert sha(blob(m['source_commit']+':'+n))==h,n
assert data('third-red/tests/test_shadow_reap_deadline.py')==blob(m['source_commit']+':tests/test_shadow_reap_deadline.py')
for n in ['src/main.c','src/runtime/shadow_runner.h','src/nanovirt/shadow_runner.c']:
 assert data('prepared386/source/'+n)==blob('386cf15309a524078bf1ad2ee36566d65ea43735:'+n)
proc.stdin.close();assert proc.wait()==0
print('PASS archive integrity and declared partial result: %d records, exact-source normal/O3 13/10/6/42 PASS; SAN13/10/6 PASS, original42 FAIL only1252; three late-exit controls PASS every profile. Full bootstrap, Stage2, paired and release acceptance remain open.'%len(rows))
