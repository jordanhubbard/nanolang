"""I verify a pure-source copy candidate's translation readiness, not execution parity."""
from pathlib import Path
import hashlib,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'059daeb734e96daac1eaec34d926b4449a0b74a36f3df57d233c03fcd43d2ac9')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
pin='abdc47ecb4758f5e608bb55406503164c3e43344'
assert m['source_commit']==pin and m['working_overlay'] and m['accepted_translation_readiness']
assert not m['accepted_component'] and not m['accepted_public_c_vm'] and not m['full_five_routes'] and not m['whole_release']
a=js('readiness/result.json');g=js('readiness/genuine-result.json');prior=js('prior-failure/result.json')
assert a['terminal'] and a['diagnostic_only'] and a['accepted_c_translation_readiness'] and a['all_guards_match']
assert g['terminal'] and g['diagnostic_only'] and g['accepted_genuine_translation_readiness'] and g['all_guards_match']
assert len(a['commands'])==7 and len(g['commands'])==6 and a['ordinary_seconds']==g['ordinary_seconds']==10
assert g['parent_sha256']==sha(data('readiness/result.json'))
assert a['parent_failure_sha256']==sha(data('prior-failure/cohort-result.json'))
assert a['source_before']==a['source_after']==g['source_before']==g['source_after']
assert a['tools_before']==a['tools_after']==g['tools_before']==g['tools_after']
assert g['hosts_before']==g['hosts_after']==prior['hosts_after']
for n,h in a['source_before'].items():assert rows['readiness/source/'+n]['sha256']==h
for n,h in {**a['tools_before'],**g['hosts_before']}.items():assert rows['external/'+n.lstrip('/')]['sha256']==h
frozen=Path(prior['producer']).parent.parent
for n,h in prior['images_after'].items():assert a['tools_before'][str(frozen/n)]==h
producer='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
assert g['producer_sha256']==g['producer_after']==prior['producer_sha256']==producer
assert a['tools_before'][g['producer']]==producer
origin=js('origin/nanoc_bootstrap.json');assert origin['phase']=='stage2' and origin['raw_stage1_stage2_equal'] and origin['artifacts']['nanoc_stage2']==producer
assert prior['bootstrap_source_pin']=='7b9f08ee0941c23810d18fa3f8ebfbb759c968de'
for c in a['commands']+g['commands']:
 assert c['exit']==0 and not c.get('timed_out')
 assert c['timeout']==(60 if c['label']=='typed-ast' else 600)
 assert rows['readiness/'+c['label']+'.stdout'] and rows['readiness/'+c['label']+'.stderr']
 if c['label'].endswith('-translate'):assert Path(c['argv'][0]).name=='nvm2c'
 elif c['label'].endswith('-genuine'):assert c['argv'][0]==g['producer']
for family in ['ownership_queries','affine_globals','affine_arrays']:
 for suffix in ['','-genuine']:
  assert rows['readiness/'+family+suffix+'.nvm']['bytes']>0
  assert rows['readiness/'+family+suffix+'.c']['bytes']>0
 assert 'narr_t versus nmap_value' in data('prior-failure/command-0005.stderr').decode()
 assert 'narr_t versus nmap_value' in data('prior-failure/command-0008.stderr').decode()
# I compare every source byte with the declared Git pin, permitting one explicit copy expression.
p=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
changed=[]
for n,h in a['source_before'].items():
 p.stdin.write((pin+':'+n+'\n').encode());p.stdin.flush();parts=p.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',n
 original=p.stdout.read(int(parts[2]));assert p.stdout.read(1)==b'\n'
 actual=data('readiness/source/'+n)
 if original!=actual:
  changed.append(n)
  needle=b'    return Query { status: 0, number: out.number, fields: (array_slice selected 0 out.number), globals: out.globals,'
  replacement=b'    let mut copied_fields: array<int> = []\n    for field_index in (range 0 out.number) { set copied_fields (array_push copied_fields (at selected field_index)) }\n    return Query { status: 0, number: out.number, fields: copied_fields, globals: out.globals,'
  assert original.count(needle)==1 and original.replace(needle,replacement)==actual,n
 assert sha(actual)==h
p.stdin.close();assert p.wait()==0 and changed==['src_nano/nvm2c/ownership_queries.nano']
draft=js('draft-missing-images/result.json');assert draft['terminal'] and len(draft['commands'])==7 and all(c['exit']==0 for c in draft['commands'])
assert 'AttributeError' in data('controller-logs/nanolang-path-copy-readiness-guarded.log').decode()
assert data('draft-missing-images/candidate.diff')==data('readiness/candidate.diff')
print('PASS %d records; 13 image-guarded typed/build/translation commands across query/global/array consumers and both producers; one pure copy expression; prior failures and observer drafts retained. No complete runtime/query/global/array/full-route/paired/compiler-repair/release acceptance.'%len(rows))
