"""I verify private body results, derived cache/facts, original limits and failed controls."""
from pathlib import Path
import hashlib,json,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'05fc35289accc12ef12b8777303fea07fb1a78f7dbc4800db5c4fd61aa223838')
sha=lambda b:hashlib.sha256(b).hexdigest()
assert m['source_commit']=='c3018e1795ad5cddf5b669cb7c80983e35067c02'
assert m['accepted_private_body_c_vm'] and not m['accepted_component'] and not m['full_five_routes'] and not m['whole_release']
a=js('audit/result.json');assert a['terminal'] and a['accepted_boundary_routes'] and a['all_guards_match'] and a['product_c_unchanged']
assert a['case_counts']==dict(original=1001,analysis=308) and len(a['commands'])==743
root=Path(m['roots']['audit'])
def mapped(n):
 p=Path(n)
 if p.is_relative_to(root):return 'audit/'+str(p.relative_to(root))
 return 'external/'+str(p).lstrip('/')
for group in ['source','tools','hosts','objects']:
 assert a[group+'_before']==a[group+'_after']
 for n,h in a[group+'_before'].items():assert rows['audit/source/'+n if group=='source' else mapped(n)]['sha256']==h,n
assert all(c['exit']==0 and not c.get('timed_out') for c in a['commands'])
for c in a['commands']:
 assert c['timeout']==(600 if c['label'].startswith(('build','c-original-')) else 60)
 if c['label'].startswith('vm-'):assert c['argv'][2]=='--' and len(c['argv'])==4
 if c['label'].startswith('c-analysis-'):assert len(c['argv'])==2
 if c['label'].startswith('c-original-'):assert len(c['argv'])<=9
for group in ['original','analysis']:
 cases=[c for c in a['cases'] if c['group']==group];raw=data('audit/c-'+group+'.reports');lines=raw.decode().splitlines();assert len(lines)==a['case_counts'][group]
 assert b''.join(data('audit/'+c['label']+'.stdout') for c in a['commands'] if c['label'].startswith('c-'+group+'-'))==raw
 spec=next(t for t in a['tests'] if t['route']=='actual-c' and t['group']==group);assert spec['failures']==0 and spec['sha256']==sha(raw)
 if group=='original':assert sha(raw)=='f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
 for case,line in zip(cases,lines):
  assert rows[mapped(case['path'])]['sha256']==case['sha256'];expected=case['expected']
  if group=='original':
   if case['name'].startswith(('global-algebra-','bounded-fifo-')):assert line==expected
   else:assert line.startswith(expected)
   continue
  if expected.get('malformed'):assert line=='D';continue
  result,flags,global_rows,cache_rows,fact_rows=line.split('|')
  ok,offset,reachable,visits,message=result.split(',',4);value=(ok=='1',int(offset),int(reachable),int(visits),message)
  assert value[0]==(message=='')
  if 'result' in expected:assert value==tuple(expected['result']),(case['name'],value)
  if 'ok' in expected:assert value[0]==expected['ok'],case['name']
  entries=[item.split('~') for item in cache_rows.split('#') if item];assert len(entries)==8
  for f,status in expected.get('statuses',{}).items():assert int(entries[int(f)][0])==status
  for key,column in [('allowed',0),('returns',1),('keys',2)]:
   for f,wanted in expected.get(key,{}).items():assert [int(n) for n in entries[int(f)][2].split('/')[column].split(',') if n]==wanted,(case['name'],key)
  count,rest=fact_rows.split(',',1);facts=[tuple(int(n) for n in item.split(',') if n) for item in rest.split(';') if item];assert len(facts)==int(count)
  if 'facts' in expected:assert facts==[tuple(item) for item in expected['facts']]
  if expected.get('sentinels'):assert facts==[(1234567,1,7,99,11)]*2
assert data('audit/vm-analysis.reports')==data('audit/c-analysis.reports')
assert sha(data('audit/vm-analysis.reports'))=='8c9b3b9a5c0c1f8ab6996c734e070b7fbb038a13318e5b44bd28a577a5cc8377'
assert b''.join(data('audit/'+c['label']+'.stdout') for c in a['commands'] if c['label'].startswith('vm-analysis-'))==data('audit/vm-analysis.reports')
assert all(t['failures']==0 for t in a['tests']) and len(a['tests'])==3
names={c['name'] for c in a['cases'] if c['group']=='analysis'}
assert all('primary-'+str(i) in names for i in range(256))
assert {'derived-callee-escape','late-global-load-cache','late-effect-cache','late-value-call-cache','reference-call-1-2-1','handled-effect-alias-1','unknown-union-exhaustion','requeue-initialization-loss','initialized-global-read','uninitialized-global-read'}<=names
initial=js('audit-initial/result.json');assert initial['terminal'] and initial['all_guards_match'] and [t['failures'] for t in initial['tests']]==[0,1]
assert initial['case_counts']==dict(original=1001,analysis=305)
assert {c['sha256'] for c in initial['cases']}<={c['sha256'] for c in a['cases']}
assert js('audit-initial/analysis-mismatches.json')[0]['name']=='array-length'
for suffix,expected in [('3vhce7_7',[1,0,1]),('dn1d_k4y',[0,0,1]),('nebxk3qn',[1,0,1]),('h5c3s8kp',[1,0,1]),('gq7w0adq',[0,0,0])]:
 label='review-'+suffix;check=js(label+'/result.json');assert check['terminal'] and check['all_guards_match'] and [c['exit'] for c in check['commands']]==expected
 for n,h in check['source_before'].items():assert rows[label+'/'+n]['sha256']==h
assert b'AGG_GET field 1 is unavailable' in data('review-dn1d_k4y/ordinary-shadows.stderr')
assert js('generator-draft/result.json')['corpus_generated'] is False and b'NameError' in data('generator-draft/stderr')
r=js('driver-ynd9vkbp/result.json');assert r['terminal'] and r['all_guards_match'] and all(c['exit']==0 for c in r['commands'])
assert sha(data('driver-ynd9vkbp/result.json'))==a['review_sha256'] and rows['driver-ynd9vkbp/preflight.nvm']['sha256']==a['nvm_sha256']
for n in ['src_nano/nvm2c/affine_analysis.nano','tests/nvm2c_affine_analysis_reference.c','tests/nvm2c_affine_analysis_driver.nano']:assert r['source_before'][n]==a['source_before'][n]
check=js('make/result.json');assert check['terminal'] and check['exit']==0 and check['all_guards_match'] and check['source_before']==check['source_after'] and check['timeout']==600 and check['ordinary_seconds']==10
assert 'test-nanolang-translator-affine-analysis-shadows' in check['argv']
for n,h in check['source_before'].items():assert rows['make/'+n]['sha256']==h,n
# I bind all frozen inputs and overlays to the actual committed implementation.
p=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in a['source_before'].items():
 p.stdin.write((m['source_commit']+':'+n+'\n').encode());p.stdin.flush();parts=p.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
 raw=p.stdout.read(int(parts[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,n
p.stdin.close();assert p.wait()==0
assert not subprocess.check_output(['git','diff','--name-only','5d809aa3d14d11b25adc2ee2ae391893ae5cf0f2',m['source_commit'],'--',':(top)src'],cwd=base)
print('PASS %d records; 743 commands, 1001 unchanged original C reports, 308 complete private C/VM result/cache/fact reports, independent control expectations, exact source/Git/tool/object/host guards, ordinary shadows/Make and retained failures. No public effect-global, full-source/sanitizer/paired/emission/cutover/release acceptance.'%len(rows))
