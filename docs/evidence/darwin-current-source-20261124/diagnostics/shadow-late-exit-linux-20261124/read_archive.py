"""I verify actual late admission without attributing an earlier refusal."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'ce964f53a1878153910d8fb957fd069c6f725bdf67cfd9d3c78d10a2fa91a02d')
q=js('actual/result.json');setup=js('actual/setup.json');helper=js('actual/helper-after.json');product=js('actual/product-result.json');sha=lambda b:hashlib.sha256(b).hexdigest()
assert m['diagnostic_only'] and not m['whole_release'] and not m['full_source_retry'] and not q['product_edits'] and q['terminal'] and q['exit']==0 and 'exception' not in q
assert q['before']==q['after'] and all(rows['actual/'+n]['sha256']==h for n,h in q['before'].items())
assert q['before']['bin/nanoc_c']==setup['compiler_sha256']
parent=json.loads((base.parent/'fresh-record-adoption-linux-20261124/manifest.json').read_text());old={r['path']:r for r in parent['files']}
assert setup['compiler_sha256']==old['journal/cold-source-normal/bin/nanoc_c']['sha256']
assert helper['sha256']==helper['after_sha256']==rows['actual/bin/nano_as_capture.so']['sha256']==old['journal/cold-source-normal/bin/nano_as_capture.so']['sha256']
assert q['parent_wchan_before_stop']=='hrtimer_nanosleep' and q['child_exit_poll_ready'] and 'State:\tT (stopped)' in q['parent_status_before_resume']
assert q['wait_condition']['exit']==0
phases=[e for e in q['events'] if e.get('event')=='cseed-shadow-phase'];s=phases[-1];completion=next(e for e in phases if e['phase']=='completion-written')
assert s['completed'] and s['reaped'] and not s['parent_timeout'] and not s['clock_failure'] and s['deadline_seconds']==1 and s['exit_code']==s['signal']==0
elapsed=completion['monotonic_seconds']-s['parent_start_seconds']+(completion['monotonic_nanoseconds']-s['parent_start_nanoseconds'])/1e9
assert elapsed>1.1 and s['child_pid']==q['child_pid'] and s['pid']==q['pid']
assert product['exit']==0 and product['sha256']==rows['actual/product']['sha256'];head=data('actual/product')[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
assert b'signal(SIGALRM,SIG_IGN)' in data('actual/foreign/delay.c') and b'pub extern fn delayed_finish' in data('actual/foreign/delay.nano')
assert 'PRIVATE ACCESS' in js('private-setup/result.json')['exception'] and 'retain compiler and assembler inputs' in js('helper-setup/result.json')['exception']
assert 'helper-setup/bin/nano_as_capture.so' not in rows
for n,h in m['tools_after'].items():assert rows['tool/'+n.lstrip('/')]['sha256']==h
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
for n in ['src/main.c','src/runtime/shadow_runner.h']:
 assert subprocess.check_output(['git','show',m['source_pin']+':'+n],cwd=repo)==data('actual/'+n)
print('PASS evidence: %d records; actual unchanged C seed accepts completion at %.6fs under1s deadline and publishes/runs native product. #1249 remains open; I assign no cause to #1229.'%(len(rows),elapsed))
