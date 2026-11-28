"""I verify unchanged failed methods and the actual pre-shadow refusal boundary."""
from pathlib import Path
import sys,json,subprocess,hashlib
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'8ce328185858ed974c60905df750b98f563cc1f9818dbba299a9d4a563b980db');sha=lambda b:hashlib.sha256(b).hexdigest();q=js('localization/result.json');borrow=js('localization/borrowed-current-producer.json');s=js('localization/setup-result.json');prefix='localization/source/'
assert m['diagnostic_only'] and not m['new_bootstrap'] and not m['production_edits'] and m['qualified_producer_reuse'] and not m['whole_release'] and m['issue']==1253
assert q['terminal'] and q['all_guards_match'] and q['tests_run']==5 and len(q['failures'])==5 and not q['errors'] and not q['skipped'] and len(q['commands'])==5
assert q['source_pin']==m['source_commit']==borrow['source_pin']==s['source_pin'] and s['admitted_setup'] and borrow['admitted'] and not borrow['new_bootstrap']
assert q['source_before']==q['source_after']==s['source_before'] and q['tools_before']==q['tools_after'] and q['hosts_before']==q['hosts_after']==borrow['immutable_hosts']
for n,h in q['source_before'].items():assert rows[prefix+n]['sha256']==h,n
for n,h in q['tools_before'].items():assert rows['tool/'+n.lstrip('/')]['sha256']==h
assert rows[prefix+'bin/nanoc_stage2']['sha256']==q['producer_sha256']==q['producer_after']==borrow['copied']['bin/nanoc_stage2']
head=data(prefix+'bin/nanoc_stage2')[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
parent=base.parent/'core-supervisor-bootstrap-linux-20261124';assert sha((parent/'manifest.json').read_bytes())=='dc9080dac206655d64267f2e3755fcd463376d83ffdc35d359225593c36421db';pm=json.loads((parent/'manifest.json').read_bytes());pr={v['path']:v for v in pm['files']};assert pm['source_commit']==m['source_commit'] and pm['original_make_pass']
for n,h in borrow['copied'].items():assert rows[prefix+n]['sha256']==pr['bootstrap/source/'+n]['sha256']==h
for n,h in borrow['immutable_hosts'].items():assert pr['bootstrap/'+str(Path(n).relative_to(borrow['qualified_bootstrap_journal']))]['sha256']==h
for i,c in enumerate(q['commands']):
 folder='localization/processes/%03d/'%i;assert c['terminal'] and c['communicate_timeout']==90 and not c.get('observed_timeout',False) and c['environment']['NANO_SHADOW_TIMEOUT_SECONDS'] is None
 assert c['argv'][0]==q['qualified_producer'] and c['returncode']==(1 if i<4 else 0)
 assert rows[folder+'stdout']['sha256']==c['stdout_sha256'] and rows[folder+'stderr']['sha256']==c['stderr_sha256']
 assert data(folder+'stderr')==b'';assert data(folder+'before/input.nano')==data(folder+'after/input.nano')
 if i<4:
  assert data(folder+'stdout')==b'I cannot lower this checked program: no main\n' and b'fn main' not in data(folder+'before/input.nano')
 else:
  assert data(folder+'stdout')==b'' and c['argv'][-2:]==['--target','c'] and c['environment']['NANO_CC']=='/bin/false'
  emitted=data(folder+'after/program');assert emitted and b'__nano_shadow_' not in emitted
 if i<2:
  name=Path(c['environment']['NANO_CFLAGS']).name;assert data(folder+'before/'+name)==data(folder+'after/'+name)
  assert (b'alarm(0)' if i==0 else b'exit(0)') in data(folder+'before/'+name)
 if i==1:assert data(folder+'before/program')==data(folder+'after/program')==b'prior output'
 elif i<4:assert folder+'after/program' not in rows
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in q['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+n+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',n;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==h,n
proc.stdin.close();assert proc.wait()==0
print('PASS integrity/localization: %d records; five unchanged original methods FAIL with stable inputs/tools. Four actual exit1 no-main refusals precede shadow execution; source-only exit0 lacks its notice. No repair, deadline change or current native/release acceptance claimed.'%len(rows))
