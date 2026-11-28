"""I verify private cache shadows and retained declaration-projection refusals."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'38d9168f866dfc6f39c97e20b522c77191ee2c9b867ab92f7fe369caef0cf456');sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['source_commit']=='0e9aa6e58dbc46ede67e1631731c17bf949467da'
assert m['source_shadows_only'] and m['component_only'] and not any(m[k] for k in ['c_oracle_only','full_five_routes','new_bootstrap','whole_release'])
def check(label,exits):
 q=js(label+'/result.json');assert q['terminal'] and q['all_guards_match'] and q['source_before']==q['source_after'] and q['tools_before']==q['tools_after']
 assert [c['exit'] for c in q['commands']]==exits
 for n,h in q['source_before'].items():assert rows[label+'/'+n]['sha256']==h
 for n,h in q['tools_before'].items():assert rows['external/'+n.lstrip('/')]['sha256']==h
 for c in q['commands']:
  assert c['timeout']==(600 if c['label']=='ordinary-shadow-module' else 60) and c['ordinary_shadow_seconds']==10
 return q
passed=check('passing-review',[0,0,0,0]);failed1=check('failed-review-1',[0,0,0,1]);failed2=check('failed-review-2',[0,0,0,1])
assert b'returned bool, expected int' in data('failed-review-1/ordinary-shadow-module.stderr')
assert b'I64_NE requires two integers' in data('failed-review-2/ordinary-shadow-module.stderr')
for n in ['src_nano/nvm2c/affine_analysis_cache.nano','tests/nvm2c_affine_analysis_cache_shadows.nano']:
 assert rows['passing-review/'+n]['sha256']==rows['failed-review-1/'+n]['sha256']==rows['failed-review-2/'+n]['sha256']
prior=js('failed-review-1/prior-common-source.json');assert prior['source_pin']=='1db292559fda291cc35b3a4d751976a3a964174b'
name='src_nano/nvm2c/ownership_queries.nano';old=data('failed-review-1/prior-common-source/'+name);new=data('passing-review/'+name)
assert old.replace(b'    return (owned.validate bytes ctx.model).status',b'    let checked: owned.Declaration = (owned.validate bytes ctx.model)\n    return checked.status')==new
assert rows['failed-review-2/'+name]['sha256']==sha(new)
name='src_nano/nvm2c/affine_facts.nano';old=data('failed-review-1/prior-common-source/'+name);new=data('passing-review/'+name)
assert old.replace(b'    if (!= (owned.validate bytes model).status 0) { return (empty) }',b'    let checked: owned.Declaration = (owned.validate bytes model)\n    if (!= checked.status 0) { return (empty) }')==new
# I retain failed reductions without assigning them an unproved cause.
for label in ['minimal-1','minimal-2','minimal-3']:
 q=js(label+'/result.json');assert q['terminal'] and q['all_guards_match'] and q['tool_before']==q['tool_after']
 assert [c['exit'] for c in q['commands']]==([1,0] if label=='minimal-3' else [1,1])
 assert all(c['timeout']==600 and c['ordinary_shadow_seconds']==10 for c in q['commands'])
assert b'I64_EQ requires two integers' in data('minimal-3/direct/stderr') and data('minimal-3/typed/stderr')==b''
assert data('minimal-3/direct/query.nano')==data('minimal-3/typed/query.nano')
assert data('minimal-3/direct/decoy.nano')==data('minimal-3/typed/decoy.nano')
a=data('minimal-3/direct/main.nano');b=data('minimal-3/typed/main.nano')
assert a.replace(b'assert (== (query.query).status 0)',b'let out: query.Outcome = (query.query) assert (== out.status 0)')==b
up=js('upstream-ordinary/result.json');assert up['terminal'] and up['all_guards_match'] and up['source_before']==up['source_after'] and up['tool_before']==up['tool_after']
assert up['source_pin']=='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8' and len(up['commands'])==2
assert all(c['exit']==0 and c['timeout']==600 and c['ordinary_shadow_seconds']==10 for c in up['commands'])
for n,h in up['source_before'].items():assert rows['upstream-ordinary/source/'+n]['sha256']==h
make=js('make-target/result.json');assert make['terminal'] and make['exit']==0 and make['all_guards_match']
assert make['source_before']==make['source_after']==passed['source_before'] and make['tool_before']==make['tool_after']
assert make['make_before']==make['make_after']==rows['make-target/Makefile.gnu']['sha256']
assert make['timeout']==600 and make['ordinary_shadow_seconds']==10 and '-o' in make['argv'] and 'nano_virt' in make['argv']
assert 'test-nanolang-translator-affine-analysis-cache-shadows' in make['argv'] and rows['make-target/products/nvm2c_affine_analysis_cache_shadows.nvm']['bytes']>0
assert m['make_source_commit']=='3043c9ef248a156efe215761a4a5da579c9b621d'
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def gitcheck(commit,name,h):
 proc.stdin.write((commit+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==h;return raw
for n,h in passed['source_before'].items():gitcheck(m['source_commit'],n,h)
for n,h in prior['source'].items():assert rows['failed-review-1/prior-common-source/'+n]['sha256']==h;gitcheck(prior['source_pin'],n,h)
for n,h in up['source_before'].items():gitcheck(up['source_pin'],n,h)
newmake=gitcheck(m['make_source_commit'],'Makefile.gnu',make['make_before'])
oldmake=subprocess.check_output(['git','show',m['source_commit']+':Makefile.gnu'],cwd=repo)
assert newmake.startswith(oldmake) and b'alarm 600' in newmake[len(oldmake):] and b'--emit-nvm' in newmake[len(oldmake):] and b'test-units:' in newmake[len(oldmake):]
proc.stdin.close();assert proc.wait()==0
print('PASS %d records; private cache ordinary imported shadows/AST/presence, exact source/tool guards, two retained declaration-projection refusals and exact typed-binding corrections with all assertions unchanged; reduced direct-failure/typed-pass controls and exact queued global/array ordinary shadows. No compiler repair, C differential, pure callee/full-route/sanitizer/paired/CFG/emission/release acceptance.'%len(rows))
