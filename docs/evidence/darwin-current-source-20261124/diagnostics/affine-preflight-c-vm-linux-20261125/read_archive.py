"""I verify preflight C/VM boundaries, unchanged body controls and retained failures."""
from pathlib import Path
import hashlib,json,subprocess,sys,re
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'e5293145b6b2663088a70aa30feba37a12687ea05a7807c58be5a2cfce36cc34')
sha=lambda b:hashlib.sha256(b).hexdigest()
assert m['source_commit']=='14fc376fb1573c5651a1fa79682ed3bedadece83'
assert m['accepted_c_component'] and m['accepted_vm_boundaries'] and not m['accepted_component'] and not m['full_five_routes'] and not m['whole_release']
a=js('audit/result.json');v=js('vm/result.json');r=js('review/result.json')
assert a['terminal'] and a['accepted_c_component'] and a['all_guards_match'] and a['observer_body_unchanged'] and a['product_c_unchanged']
assert a['case_counts']==dict(original=1001,preflight=228) and len(a['commands'])==146
root=Path(m['roots']['audit'])
def mapped(n):
 p=Path(n)
 for label in ['audit','review','vm']:
  start=Path(m['roots'][label])
  if p.is_relative_to(start):return label+'/'+str(p.relative_to(start))
 return 'external/'+str(p).lstrip('/')
for group in ['source','tools','hosts','objects']:
 assert a[group+'_before']==a[group+'_after']
 for n,h in a[group+'_before'].items():assert rows['audit/source/'+n if group=='source' else mapped(n)]['sha256']==h,n
assert all(c['exit']==0 and not c.get('timed_out') for c in a['commands'])
for c in a['commands']:
 assert c['timeout']==(600 if c['label'].startswith(('build','original-')) else 60)
 if c['label'].startswith(('preflight-','unobserved-')):
  paths=c['argv'][1:];assert len(paths)<=32 and (len(paths)==1 or sum(rows[mapped(n)]['bytes'] for n in paths)<=8192)
assert len(a['observer_body_controls'])==9
for control in a['observer_body_controls']:
 i=control['batch'];before=data('audit/preflight-%04d.stderr'%i).decode();after=data('audit/unobserved-%04d.stderr'%i).decode()
 normalize=lambda t:re.sub(r'^preflight=[01] ','',t,flags=re.M)
 assert normalize(before)==normalize(after) and sha(normalize(before).encode())==control['before_sha256']==control['after_sha256']
assert '-DNANOLANG_PREFLIGHT_OBSERVER_DISABLED' in next(c for c in a['commands'] if c['label']=='build-unobserved')['argv']
source=data('audit/source/tests/nvm2c_affine_preflight_reference.c').decode()
assert 'vm_decoded_function_free(decoded);return limit;' in source and '#define NVM_AFFINE_TEST_VISIT_LIMIT capture_preflight' in source
assert 'I reconstruct this separate query view with actual public C APIs.' in source
for group,count in [('original',1001),('preflight',228)]:
 cases=[c for c in a['cases'] if c['group']==group];assert len(cases)==count
 raw=data('audit/'+group+'.reports');lines=raw.decode().splitlines();assert len(lines)==count
 spec=next(t for t in a['tests'] if t['group']==group);assert spec['failures']==0 and spec['sha256']==sha(raw)
 if group=='original':assert sha(raw)=='f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
 for case,line in zip(cases,lines):
  assert rows[mapped(case['path'])]['sha256']==case['sha256']
  if group=='original' and not case['name'].startswith(('global-algebra-','bounded-fifo-')):assert line.startswith(case['expected'])
  else:assert line==case['expected']
assert sum(c['expected'].startswith('1|') for c in a['cases'] if c['group']=='preflight')==37
initial=js('audit-initial/result.json');second=js('audit-v2/result.json')
assert initial['terminal'] and initial['all_guards_match'] and [t['failures'] for t in initial['tests']]==[0,37]
assert second['terminal'] and second['accepted_c_component'] and second['all_guards_match'] and [t['failures'] for t in second['tests']]==[0,0]
assert {(c['group'],c['name']):c['sha256'] for c in initial['cases']}=={(c['group'],c['name']):c['sha256'] for c in a['cases']}
assert js('draft/result.json')['exit']!=0 and b'SyntaxError' in data('draft/python-syntax.stderr')
for label in ['review-initial','review']:
 check=js(label+'/result.json');assert check['terminal'] and check['all_guards_match'] and all(c['exit']==0 for c in check['commands'])
 assert check['source_before']==check['source_after'] and check['tools_before']==check['tools_after']
 for n,h in check['source_before'].items():assert rows[label+'/'+n]['sha256']==h
 assert check['commands'][-1]['timeout']==600 and check['ordinary_seconds']==10
for n,h in r['source_before'].items():assert a['source_before'][n]==h
assert v['terminal'] and v['accepted_boundary_route'] and v['complete_c_match'] and v['all_guards_match'] and v['reports']==228 and v['failures']==0
assert sha(data('audit/result.json'))==v['audit_result_sha256'] and sha(data('review/result.json'))==v['review_result_sha256']
assert rows['review/preflight.nvm']['sha256']==v['nvm_sha256']==v['nvm_after']
assert v['inputs_before']==v['inputs_after'] and v['tools_before']==v['tools_after']
for n,h in v['inputs_before'].items():assert rows[mapped(n)]['sha256']==h
old='/var/tmp/nanolang-hoisted-lists-cold-bootstrap-20261125-ia5_k8qp/source/bin/'
for n,h in v['tools_before'].items():assert rows['external/'+(old+n).lstrip('/')]['sha256']==h
assert data('vm/preflight.reports')==data('audit/preflight.reports') and sha(data('vm/preflight.reports'))==v['report_sha256']=='bf39b8692e5b95198377eef1c35683dc78404c94667024967a339cc8884b8a2c'
for c in v['commands']:assert c['exit']==0 and c['timeout']==60 and c['argv'][2]=='--'
bad=js('vm-initial/result.json');assert bad['terminal'] and bad['commands'][0]['exit']==1 and b'guest arguments after --' in data('vm-initial/vm-0000.stderr')
# I independently bind all frozen source and diagnostic overlays to the final source commit.
p=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in a['source_before'].items():
 p.stdin.write((m['source_commit']+':'+n+'\n').encode());p.stdin.flush();parts=p.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
 raw=p.stdout.read(int(parts[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,n
p.stdin.close();assert p.wait()==0
print('PASS %d records; 1001 original C reports, 228 independent C/VM preflight reports, 37 body-entry controls, executed C analyzer results unchanged with observer disabled, exact source/tool/object/host guards and retained failures. No pure body/callee/CFG/full-source/sanitizer/paired/emission/release acceptance.'%len(rows))
