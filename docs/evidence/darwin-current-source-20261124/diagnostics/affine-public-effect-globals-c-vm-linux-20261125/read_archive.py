"""I verify public C/VM checks, original bounds and staged immutable-store facts."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,records,data,js=read_archive(base,'fa176f795556843812dddd232de9828edce05e2db0eac975c928c2fe9ade3706')
sha=lambda b:hashlib.sha256(b).hexdigest()
assert m['source_commit']=='05b1dc70bb4e1474efe15bc5828b9e931da96c4d'
assert m['accepted_public_c_vm'] and not m['accepted_component'] and not m['full_five_routes'] and not m['whole_release']
a=js('audit/result.json');r=js('driver-review/result.json')
assert a['terminal'] and a['accepted_boundary_routes'] and a['all_guards_match'] and a['public_hook_unchanged'] and a['product_c_unchanged']
assert a['case_counts']==dict(original=1001,public=842) and len(a['commands'])==2485 and len(a['unhooked_controls'])==673
root=Path(m['roots']['audit'])
def mapped(name):
 p=Path(name)
 return 'audit/'+str(p.relative_to(root)) if p.is_relative_to(root) else 'external/'+str(p).lstrip('/')
for group in ['source','tools','hosts','objects']:
 assert a[group+'_before']==a[group+'_after']
 for n,h in a[group+'_before'].items():assert records['audit/source/'+n if group=='source' else mapped(n)]['sha256']==h,n
assert all(c['exit']==0 and not c.get('timed_out') for c in a['commands'])
for c in a['commands']:
 assert c['timeout']==(600 if c['label'].startswith(('build','c-original-')) else 60)
 if c['label'].startswith('vm-'):assert c['argv'][2]=='--' and len(c['argv'])==4
 if c['label'].startswith(('c-public-','unhooked-public-')):assert len(c['argv'])==2
 if c['label'].startswith('c-original-'):assert len(c['argv'])<=9

def inspect(line,expected):
 if expected.get('malformed'):assert line=='D';return
 if 'hash' in expected:assert line==str(expected['hash'])+',';return
 result,fact_rows=line.split('|');ok,offset,reachable,visits,message=result.split(',',4)
 value=(ok=='1',int(offset),int(reachable),int(visits),message);assert value[0]==(message=='')
 if 'result' in expected:assert value==tuple(expected['result']),(value,expected)
 if 'ok' in expected:assert value[0]==expected['ok']
 if 'error' in expected:assert message==expected['error']
 count,rest=fact_rows.split(',',1);facts=[tuple(int(n) for n in row.split(',') if n) for row in rest.split(';') if row];assert len(facts)==int(count)
 if 'facts' in expected:assert facts==[tuple(item) for item in expected['facts']]
 if expected.get('sentinels'):assert facts==[(1234567,1,7,99,11)]*2
for group in ['original','public']:
 cases=[c for c in a['cases'] if c['group']==group];raw=data('audit/c-'+group+'.reports');lines=raw.decode().splitlines();assert len(lines)==a['case_counts'][group]
 assert b''.join(data('audit/'+c['label']+'.stdout') for c in a['commands'] if c['label'].startswith('c-'+group+'-'))==raw
 spec=next(t for t in a['tests'] if t['route']=='actual-c' and t['group']==group);assert spec['failures']==0 and spec['sha256']==sha(raw)
 if group=='original':assert sha(raw)=='f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
 for case,line in zip(cases,lines):
  assert records[mapped(case['path'])]['sha256']==case['sha256']
  if group=='public':inspect(line,case['expected'])
  elif case['name'].startswith(('global-algebra-','bounded-fifo-')):assert line==case['expected']
  else:assert line.startswith(case['expected'])
assert len(a['tests'])==3 and all(t['failures']==0 for t in a['tests'])
assert data('audit/vm-public.reports')==data('audit/c-public.reports')
assert sha(data('audit/vm-public.reports'))=='467e494c5f087ce46c683ec33ad4b292d2c686f8a73faf8bbbb7caa9f3f6770f'
assert b''.join(data('audit/'+c['label']+'.stdout') for c in a['commands'] if c['label'].startswith('vm-public-'))==data('audit/vm-public.reports')
for c in a['unhooked_controls']:
 i=c['index'];assert data('audit/unhooked-public-%04d.stdout'%i)==data('audit/c-public-%04d.stdout'%i)
 assert sha(data('audit/c-public-%04d.stdout'%i))==c['sha256']
assert '-DNANOLANG_EFFECT_OBSERVER_DISABLED' in next(c for c in a['commands'] if c['label']=='build-unhooked')['argv']
assert r['terminal'] and r['all_guards_match'] and all(c['exit']==0 for c in r['commands']) and len(r['commands'])==5
assert sha(data('driver-review/result.json'))==a['review_sha256'] and records['driver-review/preflight.nvm']['sha256']==a['nvm_sha256']
for n,h in r['source_before'].items():assert records['driver-review/'+n]['sha256']==h and a['source_before'][n]==h
check=js('make/result.json');assert check['terminal'] and check['exit']==0 and check['all_guards_match'] and check['source_before']==check['source_after'] and check['timeout']==600 and check['ordinary_seconds']==10
assert 'test-nanolang-translator-affine-public-shadows' in check['argv']
for n,h in check['source_before'].items():assert records['make/'+n]['sha256']==h
# The three additional immutable controls reuse exactly the qualified production artifacts.
sm,srows,sdata,sjs=read_archive(base/'immutable-stores','bcad5a2d5a526a3a8779179f11781420eb26e3803175c0d30fca51dc1903c0c7')
s=sjs('immutable/result.json');assert sm['source_commit']=='d1687ded5ca551eb102b66b32a8eb4eafa6019e7'
assert s['terminal'] and s['accepted'] and s['all_guards_match'] and s['old_cases']==842 and s['combined_cases']==845
assert len(s['cases'])==3 and len(s['commands'])==8 and s['parent_sha256']==sha(data('audit/result.json'))
assert s['source_before']==s['source_after'] and s['artifacts_before']==s['artifacts_after']
for n,h in s['artifacts_before'].items():assert srows['artifacts/'+n.lstrip('/')]['sha256']==h
for n,h in s['source_before'].items():
 assert srows['source/'+n]['sha256']==h
 if n!='tests/test_nanolang_translator_affine_public.py':assert a['source_before'][n]==h
assert s['artifacts_before'][str(root/'reference')]==records['audit/reference']['sha256']
assert s['artifacts_before'][str(root/'reference-unhooked')]==records['audit/reference-unhooked']['sha256']
assert s['artifacts_before'][str(Path(a['review'])/'preflight.nvm')]==a['nvm_sha256']
for i,case in enumerate(s['cases']):
 assert srows['immutable/input-%d.bin'%i]['sha256']==case['sha256']
 for label,text in case['reports'].items():
  assert sdata('immutable/%d-%s.stdout'%(i,label)).decode()==text+'\n';inspect(text,case['expected'])
 assert len(set(case['reports'].values()))==1
for c in s['commands']:assert c['exit']==0 and c['timeout']==60
# I bind both frozen source sets to their recorded Git revisions.
p=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for pin,files in [(m['source_commit'],a['source_before']),(sm['source_commit'],s['source_before'])]:
 for n,h in files.items():
  p.stdin.write((pin+':'+n+'\n').encode());p.stdin.flush();parts=p.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
  raw=p.stdout.read(int(parts[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,n
p.stdin.close();assert p.wait()==0
old=subprocess.check_output(['git','show','dc4a9f5b3ecd4f45e32ab60380ffec24eb055199:src_nano/nvm2c/container.nano'],cwd=base)
assert data('audit/source/src_nano/nvm2c/container.nano')==old.replace(b'\nfn xor32(',b'\npub fn xor32(',1)
assert not subprocess.check_output(['git','diff','--name-only','5d809aa3d14d11b25adc2ee2ae391893ae5cf0f2',sm['source_commit'],'--',':(top)src'],cwd=base)
changed=subprocess.check_output(['git','diff','--name-only',m['source_commit'],sm['source_commit']],cwd=base).decode().splitlines()
assert set(changed)=={'docs/ROADMAP.md','tests/test_nanolang_translator_affine_public.py'}
print('PASS %d records; 2493 commands, 1001 original C reports, 842+3 complete public C/VM reports, 675 unhooked public controls, immutable requeue/staged rollback, exact source/Git/artifact/tool/object/host guards and real shadow Make. No full-source/sanitizer/paired/profile/emission/cutover/release acceptance.'%(len(records)+len(srows)))
