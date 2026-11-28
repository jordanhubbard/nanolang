"""I verify an interrupted qualification without manufacturing its terminal result."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'bd478577216e932bcfd42a4be08814888d4a67f913b5df69991325321a19d4f2')
sha=lambda b:hashlib.sha256(b).hexdigest()
q=js('comparison/cohort-result.json');r=js('recovery/interruption-observation.json')
assert m['interrupted'] and not m['accepted_component'] and not m['whole_release']
assert m['source_commit']==q['source_pin']=='b43042ec0121c8c43f57c1205151ad1aaba1847b'
assert not q['terminal'] and 'tests' not in q and 'all_guards_match' not in q
assert len(q['commands'])==1867 and sum('exit' in c for c in q['commands'])==1866
assert all(c['exit']==0 for c in q['commands'][:-1]) and 'exit' not in q['commands'][-1]
assert r['old_receipt_sha256']==sha(data('comparison/cohort-result.json'))
assert r['prior_terminal'] is False and r['accepted_component'] is False and not r['processes']
assert r['unfinished']==[q['commands'][-1]] and r['boot_id']
log=data('comparison/permanent-test.stderr').decode()
assert log==r['completed_method_log'] and log.count(' ... ok\n')==2 and log.endswith(' ... ')
assert not any(k.endswith('_after') for k in q if k in ['source_after','images_after','tools_after','hosts_after'])
old=Path(m['roots']['comparison'])
def mapped(name):
 p=Path(name)
 return 'comparison/'+str(p.relative_to(old)) if p.is_relative_to(old) else 'external/'+str(p).lstrip('/')
for group in ['source','images','tools','hosts']:
 assert r['post_reboot_guards'][group]['match']
 assert r['post_reboot_guards'][group]['expected']==r['post_reboot_guards'][group]['observed']==q[group+'_before']
 for n,h in q[group+'_before'].items():
  path='comparison/source/'+n if group in ['source','images'] else mapped(n)
  assert rows[path]['sha256']==h,path
assert r['post_reboot_guards']['objects']['match']
for n,h in q['object_inputs'].items():assert rows[mapped(n)]['sha256']==h
setup=js('comparison/result.json');origin=js('origin/bootstrap-result.json')
assert setup['producer_sha256']==q['producer_sha256']=='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
assert origin['terminal'] and origin['original_make_pass'] and origin['source_pin']==q['producer_source_pin']
assert sha(data('origin/bootstrap-result.json'))==setup['bootstrap_result_sha256']
assert rows[mapped(setup['producer'])]['sha256']==q['producer_sha256']
cohorts=[]
for n in rows:
 if n.startswith('comparison/tmp/') and n.endswith('/manifest.json'):
  spec=js(n);folder=n.rsplit('/',1)[0];cases=spec['corpus'];baseline=data(folder+'/actual-c.reports')
  assert len(baseline.decode().splitlines())==len(cases)
  for case,text in zip(cases,baseline.decode().splitlines()):
   assert rows[mapped(case['path'])]['sha256']==case['sha256']
   if case['name'].startswith(('global-algebra-','bounded-fifo-')):assert text==case['expected']
   else:assert text.startswith(case['expected'])
  for route in spec['routes']:
   raw=data(folder+'/'+route['label']+'.reports')
   assert raw==baseline and sha(raw)==route['sha256'] and route['reports']==len(cases)
  cohorts.append((len(cases),len(spec['routes'])))
assert sorted(cohorts)==[(336,6),(1001,6),(3598,1)]
# I compare source bytes with Git rather than trusting copied identity fields.
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in q['source_before'].items():
 proc.stdin.write((q['source_pin']+':'+n+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split()
 assert len(parts)==3 and parts[1]==b'blob',n
 raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,n
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; 1866 recorded successful commands, 2 complete methods, 1 unfinished command, preserved nonterminal receipt and post-reboot identities. No complete operand/scalar/local/five-method or release acceptance.'%len(rows))
