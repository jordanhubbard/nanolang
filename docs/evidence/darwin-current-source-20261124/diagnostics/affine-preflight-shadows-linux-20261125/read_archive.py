"""I verify ordinary preflight shadows without claiming a converged function."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'51376fecfba8d0318206e6e26e5703a302f86b61bca106bc347c282327c8250f')
assert m['source_commit']=='274f605201b90625afbdcbec608ec5e8defbce74'
assert not m['accepted_component'] and not m['full_five_routes'] and not m['whole_release']
q=js('make/result.json');assert q['terminal'] and q['exit']==0 and q['all_guards_match']
assert q['source_before']==q['source_after'] and q['timeout']==600 and q['ordinary_seconds']==10
assert q['argv'][5]=='test-nanolang-translator-affine-preflight-shadows' and q['argv'][4]=='nano_virt'
for n,h in q['source_before'].items():assert rows['make/'+n]['sha256']==h,n
assert rows['make/products/nvm2c_affine_preflight_shadows.nvm']['bytes']>0
assert 'test-units: test-nanolang-translator-affine-preflight-shadows' in data('make/Makefile.gnu').decode()
for suffix in ['l2nvvrd9','qkefld7n','h1akr998','1paekhu_']:
 label='review-'+suffix+'/'
 r=js(label+'result.json');assert r['terminal'] and r['all_guards_match']
 assert r['source_before']==r['source_after'] and r['tools_before']==r['tools_after'] and r['ordinary_seconds']==10
 for n,h in r['source_before'].items():assert rows[label+n]['sha256']==h
 assert [c['exit'] for c in r['commands']]==([1,0,1] if suffix=='l2nvvrd9' else [0,0,0])
 assert r['commands'][-1]['timeout']==600
 for n,h in r['tools_before'].items():
  path='/var/tmp/nanolang-hoisted-lists-cold-bootstrap-20261125-ia5_k8qp/source/bin/'+n
  assert rows['external/'+path.lstrip('/')]['sha256']==h
assert 'context_sample' in data('review-l2nvvrd9/typed-ast.stderr').decode()
first=js('control/initial-make-failure.json');assert first['terminal'] and not first['compiler_started'] and first['phase']=='preparation-copy'
# I bind the exact new target, all pure dependencies and external test entry to source Git objects.
p=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in q['source_before'].items():
 if not (n=='Makefile.gnu' or n.startswith('src_nano/') and n.endswith('.nano') or n=='tests/nvm2c_affine_preflight_shadows.nano'):continue
 p.stdin.write((m['source_commit']+':'+n+'\n').encode());p.stdin.flush();parts=p.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
 raw=p.stdout.read(int(parts[2]));assert p.stdout.read(1)==b'\n' and hashlib.sha256(raw).hexdigest()==h,n
p.stdin.close();assert p.wait()==0
print('PASS %d records; exact pure preflight/imported shadows, typed AST/presence, real Make target, source/tool guards, retained draft failures and Git inputs. No unchanged-C differential, whole-walk, caller/effect convergence, paired/sanitizer or release acceptance.'%len(rows))
