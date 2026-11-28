"""I verify actual process controls without attributing the full-source refusal."""
from pathlib import Path
import sys,subprocess,hashlib,json
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'1680a82ad813d393f4db389b1c3757dba654218f0d23e26efe7ce42035339367')
sha=lambda b:hashlib.sha256(b).hexdigest();q=js('journal/result.json');c=js('journal/config.json')
assert m['diagnostic_only'] and q['diagnostic_only'] and c['diagnostic_only'] and not m['full_source_retry'] and not m['whole_release'] and not q['production_changes'] and q['terminal']
assert all(r['exit']==0 for r in q['commands']) and len(q['observations'])==3
assert rows['journal/control.c']['sha256']==q['source_sha256'] and rows['journal/control']['sha256']==q['image_sha256']
head=data('journal/control')[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
for n,h in m['tools'].items():assert rows['tool/'+n.lstrip('/')]['sha256']==h
source=data('journal/source-main.c');assert sha(source)==c['source_main_sha256']
text=source.decode();start=text.index('    struct timespec start, now, pause = {0, 10000000};',text.index('static bool check_interpreted_shadows'));end=text.index('    int wait_error = errno;',start);loop=text[start:end].encode();assert loop==data('journal/original-parent-loop.inc') and sha(loop)==c['parent_loop_sha256']
control=data('journal/control.c');assert b'#define waitpid observed_waitpid\n#include "original-parent-loop.inc"\n#undef waitpid' in control and b'WEXITED|WNOHANG|WNOWAIT' in control
assert b'signal(SIGALRM,SIG_IGN)' in control and b'delay(1300)' in control
for mode,o in enumerate(q['observations']):
 assert o==json.loads(data('journal/'+str(mode)+'.stdout')) and o['mode']==mode and o['deadline_seconds']==1 and o['receipt'] and o['exit_code']==0
 assert o['parent_timeout']==(mode!=0) and o['waited_pid']>0
 if mode:
  assert o['observer_pid']==o['waited_pid'] and o['observer_zombie_seconds']<o['reaped_seconds']
  assert (o['child_before_exit_seconds']<1)==(mode==1) and (o['observer_zombie_seconds']<1)==(mode==1)
assert js('journal/initial-build/build-result.json')['commands'][0]['exit']!=0 and b'misleading-indentation' in data('journal/initial-build/build.stderr')
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();assert subprocess.check_output(['git','show',m['source_pin']+':src/main.c'],cwd=repo)==source
print('PASS integrity: %d records; ordinary success + independently observed timely/late clean exits with delayed parent checks; original loop/deadline retained. I assign no cause to the full-source refusal.'%len(rows))
