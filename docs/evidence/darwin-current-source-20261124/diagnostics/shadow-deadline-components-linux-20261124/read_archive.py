"""I verify source/VM component gates while retaining unmet original Stage2 gates."""
from pathlib import Path
import sys,json,subprocess,hashlib,re
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'0cf805018fde466b9d908e328fc01bab1fe3216b2ab22e463a81de78f7c22b5b');sha=lambda b:hashlib.sha256(b).hexdigest();q=js('component/cold-matrix-result.json');normal=js('normal-required-wrapper/normal-component-postconditions.json');failed=js('normal-required-wrapper/cold-matrix-result.json');initial=js('initial-output-setup/cold-matrix-result.json')
assert m['component_accepted'] and not m['required_native_stage2_complete'] and not m['whole_release'] and not m['full_source_retry']
assert q['terminal'] and q['component_accepted'] and not q['required_stage2_complete'] and failed['terminal'] and initial['terminal']
assert not failed.get('accepted',False) and not initial.get('accepted',False) and normal['original_wrapper_exit']==2 and not normal['required_stage2_complete']
assert q['candidate']==failed['candidate']==m['candidate'] and q['pin']==failed['pin']==m['source_base']
assert b'FileNotFoundError' in data('normal-required-wrapper/cold-normal-focused.stderr') and b'bin/nanoc_stage2' in data('normal-required-wrapper/cold-normal-focused.stderr') and b'FAILED (errors=22)' in data('normal-required-wrapper/cold-normal-focused.stderr')
assert b'FileNotFoundError' in data('initial-output-setup/cold-normal-focused.stderr') and b'.test_output/shadow-completion-' in data('initial-output-setup/cold-normal-focused.stderr')
assert q['tools']==q['tools_after']==normal['tools_after']
for n,h in q['tools'].items():assert rows['tool/'+n.lstrip('/')]['sha256']==h
profiles=[(normal['normal_profile'],'normal-required-wrapper'),*((p,'component') for p in q['profiles'])]
assert [p['mode'] for p,label in profiles]==['normal','san','production']
for p,label in profiles:
 mode=p['mode'];prefix=label+'/cold-source-'+mode+'/';assert p['build']['exit']==0 and not p['provider_copies']
 after=normal['source_after'] if mode=='normal' else p['after'];assert p['before']==after
 if mode!='normal':assert p['focused']['exit']==0
 for n,h in p['before'].items():assert rows[prefix+n]['sha256']==h
 for n,h in q['candidate'].items():assert p['before'][n]==h
 assert rows[prefix+'bin/nanoc_c']['sha256']==p['image_sha256']==rows[label+'/cold-'+mode+'-before-focused-image']['sha256']
 for name in ['nanoc_c','nano_vm','nvm2c','nanoisa']:
  head=data(prefix+'bin/'+name)[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
 stderr=data(label+'/cold-'+mode+'-focused.stderr')
 assert re.search(rb'Ran 12 tests in [^\n]+\n\nOK\n',stderr) and re.search(rb'Ran 10 tests in [^\n]+\n\nOK\n',stderr) and re.search(rb'Ran 6 tests in [^\n]+\n\nOK\n',stderr)
 cases=[n for n in rows if n.startswith(prefix+'.test_output/shadow-reap-') and n.endswith('/process.json')];assert len(cases)==2
 for n in cases:
  process=js(n);assert process['returncode']==1 and process['deadline_seconds']==1
  folder=n.rsplit('/',1)[0];error=data(folder+'/stderr');assert b'after 1 seconds' in error
  if n.endswith('/cseed/process.json'):
   assert data(folder+'/product')==b'I retain the prior output after late completion.\n'
   events=[json.loads(l) for l in error.splitlines() if l.startswith(b'{"event":"cseed-shadow-phase"')];s=events[-1];e=next(e for e in events if e['phase']=='completion-written');elapsed=e['monotonic_seconds']-s['parent_start_seconds']+(e['monotonic_nanoseconds']-s['parent_start_nanoseconds'])/1e9
   assert elapsed>1 and s['parent_timeout'] and s['completed'] and s['reaped'] and s['exit_code']==s['signal']==0 and not s['clock_failure']
red=data('red/red.stderr');assert b'Ran 2 tests' in red and b'FAILED (failures=2)' in red and b'AssertionError: 0 == 0' in red
assert js('red/red-result.json')['exit']==1 and js('component/sanitizer-negative-case-audit.json')['unexpected_sanitizer_reports']==[]
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(spec):
 proc.stdin.write((spec+'\n').encode());proc.stdin.flush();h=proc.stdout.readline().split();assert h[1]==b'blob',spec;b=proc.stdout.read(int(h[2]));assert proc.stdout.read(1)==b'\n';return b
assert set(subprocess.check_output(['git','diff','--name-only',m['source_base'],m['source_commit']],cwd=repo,text=True).splitlines())==set(q['candidate'])
for n,h in profiles[0][0]['before'].items():assert sha(blob(m['source_commit']+':'+n))==h
assert data('red/tests/test_shadow_reap_deadline.py')==blob(m['source_commit']+':tests/test_shadow_reap_deadline.py')
for n in ['src/main.c','src/runtime/shadow_runner.h']:assert data('red/'+n)==blob('63248db641ccca41bfa5ccfb1bff0d1b32574d39:'+n)
proc.stdin.close();assert proc.wait()==0
print('PASS integrity: %d records; two original late-admission regressions retained; original available12/10/6 source+VM cohorts pass normal/SAN/O3. Required Stage2 wrapper remains FAIL/missing provider; full-source/paired/release acceptance remains open.'%len(rows))
