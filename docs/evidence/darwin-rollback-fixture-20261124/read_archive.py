"""I verify my retained Darwin terminal and bounded fixture red/green receipt."""
from pathlib import Path
import hashlib,json,subprocess,tarfile
here=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((here/'manifest.json').read_text())
for n,v in manifest.items():
 data=(here/n).read_bytes();assert len(data)==v['bytes'] and sha(data)==v['sha256'],n
with tarfile.open(here/'terminal.tar.gz') as t:
 get=lambda n:t.extractfile(n).read()
 q=json.loads(get('result.json'));cfg=json.loads(get('config.json'));guards=json.loads(get('terminal-guards.json'));old=get('failed-fixture.c')
 assert q['terminal'] and q['controller_exit']==2
 assert q['tools_before']==q['tools_at_terminal']
 assert guards['guards']==cfg['sources']==q['profiles'][0]['before']
 assert guards['source_guards_match'] and guards['tool_guards_match']
 rows={v['label']:v for v in q['commands']}
 assert rows['normal-build']['exit']==rows['normal-tool-controls']['exit']==0
 assert rows['normal-focused']['exit']==2 and not rows['normal-focused']['timed_out']
 assert len(q['profiles'])==1 and not q['source_stages']
 err=get('normal-focused.stderr').decode()
 assert err.count("error: call to undeclared function 'asprintf'")==2
 assert err.count("error: call to undeclared function 'mkdtemp'")==1
 assert 'Ran 20 tests' in err and '\nOK\n' in err
 assert old==subprocess.check_output(['git','show',cfg['pin']+':tests/test_module_rejection_rollback.c'],cwd=here)
inputs=''.join(cfg['pin']+':'+n+'\n' for n in cfg['sources']).encode()
batch=subprocess.run(['git','cat-file','--batch'],input=inputs,capture_output=True,check=True,cwd=here).stdout
cursor=0
for name,expected in cfg['sources'].items():
 end=batch.index(b'\n',cursor);header=batch[cursor:end].split();assert header[1]==b'blob';size=int(header[2]);data=batch[end+1:end+1+size];cursor=end+size+2;assert sha(data)==expected,name
assert cursor==len(batch)
with tarfile.open(here/'fixture-darwin.tar.gz') as t:
 get=lambda n:t.extractfile(n).read()
 d=json.loads(get('result.json'));fixed=get('tests/test_module_rejection_rollback.c')
 assert [v['exit'] for v in d['commands']]==[1,0,0] and d['terminal'] and d['source_guards_match']
 assert d['source_sha256']==sha(fixed)
 assert fixed.replace(b'#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)\n#define _DARWIN_C_SOURCE\n#endif\n',b'',1)==old
 red,green=d['commands'][:2]
 # My observer retained a shared argv list: its red row now contains the green source path.
 assert red['args']==green['args']
 assert b'tests/test_module_rejection_rollback.c:37:33: error:' in get('red.stderr')
 assert b'undeclared function' in get('red.stderr')
l=json.loads((here/'fixture-linux-result.json').read_text());assert l['terminal'] and l['source_sha256']==sha(fixed)
assert [v['mode'] for v in l['profiles']]==['normal','san','production']
for p in l['profiles']:assert [v['exit'] for v in p['commands']]==[0,0]
print('PASS: %d Git source guards, unchanged Darwin terminal, fixture-only delta, Darwin red/green and Linux normal/SAN/O3.'%len(cfg['sources']))
