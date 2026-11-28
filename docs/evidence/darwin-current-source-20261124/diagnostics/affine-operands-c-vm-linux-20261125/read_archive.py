"""I verify unchanged frame reports and the complete new primitive matrix."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'df5b9f09a09f574cd46ea79a90968114b11154b15d9ee32ce329199686f2a0ff');sha=lambda raw:hashlib.sha256(raw).hexdigest();s=js('comparison/result.json');q=js('comparison/compare-result.json');n=js('comparison/nvm-result.json');light=js('comparison/light-result.json')
assert m['component_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and s['working_overlay'] and s['source_producer_not_selected'] and s['hosts_before']==s['hosts_after']=={}
assert q['terminal'] and q['passed'] and q['source_guards_match'] and q['image_guards_match'] and len(q['commands'])==1152 and q['original_cases']==1001 and q['new_cases']==3598
assert n['terminal'] and n['passed_nvm'] and [v['exit'] for v in n['commands']]==[1,0,0]
assert b'I do not provide an array value or call ABI' in data('comparison/source-only.stderr') and b'I do not provide an array value or call ABI' in data('comparison/baseline-checked-c.stderr')
assert light['commands'][0]['exit']==0 and light['commands'][1]['exit']==1 and light['objects_unchanged'] and light['source_unchanged']
for name,h in s['source_after'].items():assert rows['comparison/source/'+name]['sha256']==h
for name,h in s['images_after'].items():assert rows['comparison/source/'+name]['sha256']==h
for name,h in {**s['tools_after'],**light['object_inputs']}.items():assert rows['external/'+name.lstrip('/')]['sha256']==h
for row in q['commands']:assert row['exit']==0 and row['timeout']==(600 if row['group']=='original' else 60)
for group,count,wanted in [('original',1001,'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'),('operands',3598,'7fd4b2b99d26deb52ce793583a884f294d2ebd135e1c3cce094e7a0af25e731b')]:
 c=data('comparison/'+group+'-actual-c.reports');v=data('comparison/'+group+'-nano-vm.reports');assert c==v and sha(c)==wanted;reports=c.decode().splitlines();assert len(reports)==count;cases=[r for r in q['cases'] if r['group']==group];assert len(cases)==count
 for case,actual in zip(cases,reports):
  assert rows['comparison/'+case['path']]['sha256']==case['sha256'] and actual.startswith(case['expected'])
  if case['expected'] in ['D','0','1']:assert actual==case['expected']
 for label in ['actual-c','nano-vm']:
  emitted=b''.join(data('comparison/compare-%04d.stdout'%r['number']) for r in q['commands'] if r['group']==group and r['route']==label);assert emitted==c
assert len([c for c in q['cases'] if c['name'].startswith('profile-') and c['name'].split('-')[1].isdigit()])==512
assert js('comparison/exact-policy-refusal-audit.json')['passed'] and js('comparison/integration-audit.json')['original_python_methods_and_corpus_byte_identical'] and js('comparison/integration-audit.json')['all_original_make_recipes_byte_identical']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(pin,name):
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in s['source_before'].items():
 if name in ['docs/ROADMAP.md','tests/test_nanolang_translator_affine_frame.py']:continue
 assert sha(blob(m['source_commit'] if name in s['overlays'] else s['source_base'],name))==h,name
latest=blob(m['source_commit'],'tests/test_nanolang_translator_affine_frame.py');assert latest==data('comparison/latest-test.py') and latest.startswith(blob(s['source_base'],'tests/test_nanolang_translator_affine_frame.py'))
assert blob(m['source_commit'],'Makefile.gnu').startswith(blob(s['source_base'],'Makefile.gnu'))
for name in ['src_nano/nvm2c/affine_frame.nano','tests/nvm2c_affine_frame_driver.nano']:
 old=[v for v in blob(s['source_base'],name).splitlines() if b'assert ' in v];new=iter(v for v in blob(m['source_commit'],name).splitlines() if b'assert ' in v);assert all(any(v==line for v in new) for line in old)
proc.stdin.close();assert proc.wait()==0
print('PASS %d records:1152 C/VM commands, original1001 byte-identical and3598 new reports including512 opcode modes; exact policy/refusal expectations and prior assertions/recipes retained. Checked-C array-profile refusals preserved. Full-five/sanitizers/current compiler, transfer/CFG/C emission/cutover/paired/release remain open.'%len(rows))
