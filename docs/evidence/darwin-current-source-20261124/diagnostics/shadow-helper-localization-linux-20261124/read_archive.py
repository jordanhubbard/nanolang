"""I verify unchanged failed methods and the actual pre-shadow refusal boundary."""
from pathlib import Path
import sys,json,subprocess,hashlib
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'9a3f8e6c4ed3ea22c54c56cf1e3cc4a3541b0433bb6acabe49cefc4bfda70562');sha=lambda b:hashlib.sha256(b).hexdigest();q=js('localization/result.json');borrow=js('localization/borrowed-current-producer.json');s=js('localization/setup-result.json');prefix='localization/source/'
assert m['diagnostic_only'] and not m['new_bootstrap'] and not m['production_edits'] and m['qualified_producer_reuse'] and not m['whole_release'] and m['issue']==1253
assert q['terminal'] and q['all_guards_match'] and q['tests_run']==2 and len(q['failures'])==2 and not q['errors'] and not q['skipped'] and len(q['commands'])==2
assert q['source_pin']==m['source_commit']==borrow['source_pin']==s['source_pin'] and s['admitted_setup'] and borrow['admitted'] and not borrow['new_bootstrap']
assert q['source_before']==q['source_after']==s['source_before'] and q['tools_before']==q['tools_after'] and q['hosts_before']==q['hosts_after']==borrow['immutable_hosts']
for n,h in q['source_before'].items():assert rows[prefix+n]['sha256']==h,n
for n,h in q['tools_before'].items():assert rows['tool/'+n.lstrip('/')]['sha256']==h
assert rows[prefix+'bin/nanoc_stage2']['sha256']==q['producer_sha256']==q['producer_after']==borrow['copied']['bin/nanoc_stage2']
head=data(prefix+'bin/nanoc_stage2')[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
parent=base.parent/'shadow-entry-cold-bootstrap-linux-20261124';assert sha((parent/'manifest.json').read_bytes())=='b3c85f73568c2c21784de9df63b80b68f9df1e296abdafab91e983de3559ae15';pm=json.loads((parent/'manifest.json').read_bytes());pr={v['path']:v for v in pm['files']};assert pm['source_commit']==m['source_commit'] and pm['original_make_pass']
for n,h in borrow['copied'].items():assert rows[prefix+n]['sha256']==pr['bootstrap/source/'+n]['sha256']==h
for n,h in borrow['immutable_hosts'].items():assert pr['bootstrap/'+str(Path(n).relative_to(borrow['qualified_bootstrap_journal']))]['sha256']==h
assert q['producer_images_before']==q['producer_images_after']==borrow['copied']
assert m['original_methods']==m['failures']==2 and m['source_commit']=='16b081fdf472dfe233ebf5975f149f6aa9d33783' and borrow['no_stage_alias']
old=json.loads((base.parent/'native-shadow-localization-linux-20261124/manifest.json').read_bytes());oldrows={v['path']:v for v in old['files']}
for i,c in enumerate(q['commands']):
 folder='localization/processes/%03d/'%i;assert c['terminal'] and c['communicate_timeout']==90 and not c.get('observed_timeout',False) and c['environment']['NANO_SHADOW_TIMEOUT_SECONDS'] is None
 assert c['argv'][0]==q['qualified_producer'] and c['returncode']==1 and c['environment']['NANO_CC'] is None
 assert rows[folder+'stdout']['sha256']==c['stdout_sha256'] and rows[folder+'stderr']['sha256']==c['stderr_sha256'] and data(folder+'stderr')==b''
 symbol='cancel_shadow_alarm' if i==0 else 'leave_shadows_early';line=2 if i==0 else 3
 assert data(folder+'stdout')==('I cannot lower the selected shadows: I cannot lower shadow f at merged line %d: unsupported extern result or symbol %s\n'%(line,symbol)).encode()
 for name in ['input.nano',Path(c['environment']['NANO_CFLAGS']).name]:
  assert data(folder+'before/'+name)==data(folder+'after/'+name)
  assert rows[folder+'before/'+name]['sha256']==oldrows['localization/processes/%03d/before/'%i+name]['sha256']
 assert b'fn main' not in data(folder+'before/input.nano')
 if i==0:assert folder+'before/program' not in rows and folder+'after/program' not in rows
 else:assert data(folder+'before/program')==data(folder+'after/program')==b'prior output'
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in q['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+n+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',n;raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==h,n
proc.stdin.close();assert proc.wait()==0
print('PASS integrity/localization: %d records; two unchanged current16b Stage2 methods FAIL on unsupported helper symbols before execution; source/images/tools/hosts and prior product preserved. No repair or full native/release acceptance.'%len(rows))
