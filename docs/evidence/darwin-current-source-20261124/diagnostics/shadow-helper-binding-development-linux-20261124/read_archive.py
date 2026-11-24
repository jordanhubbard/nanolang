"""I verify both development attempts without claiming a new bootstrap."""
from pathlib import Path
import sys,json,hashlib,subprocess,re,ast
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'69f7114195739a838f645eb0e7cce5469fc84f3fd86b85583c5a7770e9caf724')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['development_only'] and not m['new_bootstrap'] and not m['stage_alias'] and not m['whole_release'] and not m['shared_library_semantic_equivalence']
assert m['ordinary_shadow_seconds']==10
coldbase=base.parent/'shadow-entry-cold-bootstrap-linux-20261124'
cm,cr,cd,cj=read_archive(coldbase,'b3c85f73568c2c21784de9df63b80b68f9df1e296abdafab91e983de3559ae15')
cold=cj('bootstrap/bootstrap-result.json')
assert cold['terminal'] and cold['original_make_pass'] and cold['exit']==0
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(pin,name):
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split()
 assert len(header)==3 and header[1]==b'blob',(pin,name)
 raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';return raw
for prefix in ['first-producer','revised-producer']:
 p=js(prefix+'/result.json');seed=js(prefix+'/compiler-seed-result.json');native=js(prefix+'/development-native-result.json')
 assert p['terminal'] and p['probe_pass'] and p['all_guards_match']
 assert seed['terminal'] and seed['exit']==0 and seed['development_seed_accepted'] and seed['timeout']==600 and seed['ordinary_seconds']==10
 assert native['terminal'] and native['development_native_accepted'] and native['ordinary_seconds']==10
 assert [c['label'] for c in native['commands']]==['development-verify','development-dump','development-translate','development-native','development-help']
 assert all(c['exit']==0 and c['timeout']==600 and not c.get('timed_out') for c in native['commands'])
 assert native['input_module']==seed['product']
 for report in [p,seed,native]:
  assert report['source_base']==m['source_base'] and not report['new_bootstrap'] and not report['stage_alias'] and report['all_guards_match']
  for group in ['source','images','tools']:
   assert report[group+'_before']==report[group+'_after']
  for n,h in report['source_before'].items():
   assert rows[prefix+'/source/'+n]['sha256']==h
   assert h==report['overlays'].get(n,sha(blob(m['source_base'],n)))
  for n,h in report['images_before'].items():assert rows[prefix+'/source/'+n]['sha256']==h
  for group in ['tools_before','hosts_before']:
   for n,h in report.get(group,{}).items():assert rows['external/'+n.lstrip('/')]['sha256']==h
 for n,v in native['products'].items():assert rows[prefix+'/products/'+n]['sha256']==v['sha256'] and rows[prefix+'/products/'+n]['bytes']==v['bytes']
 assert native['hosts_before']==native['hosts_after']
 for name,h in p['images_before'].items():assert h==cr['bootstrap/source/'+name]['sha256']
 for name,h in cold['source_after'].items():
  if name.endswith(('.c','.h','.inc')):assert h==p['source_before'][name]
 if prefix=='revised-producer':
  for n,h in p['overlays'].items():assert sha(blob(m['source_commit'],n))==h
for prefix in ['first-controls','revised-controls']:
 r=js(prefix+'/result.json');s=js(prefix+'/setup-result.json')
 assert r['terminal'] and s['terminal'] and s['admitted'] and r['all_guards_match']
 assert r['development_compiler'] and not r['new_bootstrap'] and not r['stage_alias'] and not r['whole_release']
 assert r['compiler'].endswith('/bin/nanoc_shadow_development') and r['compiler_sha256']==r['compiler_after']
 assert rows[prefix+'/source/bin/nanoc_shadow_development']['sha256']==r['compiler_sha256']
 for group in ['source','images','tools','hosts']:
  assert r[group+'_before']==r[group+'_after']
  for n,h in r[group+'_before'].items():
   key=prefix+'/source/'+n if group in ['source','images'] else 'external/'+n.lstrip('/')
   assert rows[key]['sha256']==h
 for c in r['commands']:
  assert c['terminal'] and not c.get('observed_timeout')
  if c['argv'][0]==r['compiler']:assert c['communicate_timeouts']==[90]
 if prefix=='first-controls':
  assert not r['accepted'] and r['tests_run']==50 and len(r['failures'])==9 and len(r['errors'])==2
  assert b'Undefined function symbol: foreign_value' in data('outside/first-controls.stderr')
 else:
  assert r['accepted'] and r['tests_run']==36 and not r['failures'] and not r['errors'] and not r['skipped']
  for n,h in r['overlays'].items():assert sha(blob(m['source_commit'],n))==h
  log=data(prefix+'/run.py.err');assert b'Ran 36 tests' in log and log.rstrip().endswith(b'OK')
  for method,needle in [('cancel_shadow_alarm',b'after 10 seconds'),('leave_shadows_early',b'shadow')]:
   found=[]
   for i,c in enumerate(r['commands']):
    if c['argv'][0]!=r['compiler']:continue
    pre=prefix+'/processes/%04d/before/input.nano'%i
    if pre in rows and method.encode() in data(pre):found.append((i,c))
   assert len(found)==1,(method,found)
   i,c=found[0];assert c['exit']!=0 and needle in data(prefix+'/processes/%04d/stderr'%i)
   if method=='cancel_shadow_alarm':assert 10<=c['seconds']<90
  receipts=[n for n in rows if n.startswith('revised-phase-observations/') and n.endswith('/receipt.json')]
  assert len(receipts)==8
  for name in receipts:
   receipt=js(name);assert Path(receipt['tmpdir']).is_relative_to(Path(m['roots']['revised-controls'])/'tmp')
   for path,h in receipt['files'].items():assert rows[name.rsplit('/',1)[0]+'/'+path]['sha256']==h
  assert len([n for n in rows if n.startswith('revised-phase-observations/') and n.endswith('/helpers.so')])==8
 for name in ['tests/test_native_shadows.py','tests/test_shadow_runner.py','tests/test_make_timeouts.py','tests/test_native_shadow_only_entry.py']:
  assert data(prefix+'/source/'+name)==blob(m['source_base'],name)
assert not any(n.endswith('/source/bin/nanoc_stage2') or n.endswith('/source/bin/nanoc_bootstrap.json') for n in rows)
proc.stdin.close();assert proc.wait()==0
print('PASS: %d archived records; first development attempt FAIL retained, revised native36 PASS. No new bootstrap, current Stage2, paired, pure-translator or release acceptance.'%len(rows))
