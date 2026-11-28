"""I verify scalar/frame reports, preserved failures and test-helper sanitizer cleanup."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'e935d8db72480bd294dc34eceec760bd627f9f5d33b8716a13472f6878060a1c')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('comparison/result.json');q=js('comparison/light-result.json')
assert m['source_commit']==s['source_base']=='9d5ace7053b6904ec7dd3f98225b4fa4b72cb998'
assert m['component_only'] and m['c_vm_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release'])
assert q['terminal'] and q['passed_c_vm'] and q['all_guards_match'] and len(q['commands'])==3837 and q['ordinary_seconds']==10

def absolute_record(name):
 path=Path(name)
 for label,root in sorted(m['roots'].items(),key=lambda pair:len(pair[1]),reverse=True):
  if path.is_relative_to(root):return label+'/'+str(path.relative_to(root))
 return 'external/'+str(path).lstrip('/')

def verify_selected(label):
 setup=js(label+'/result.json');trial=js(label+'/light-result.json')
 assert setup['terminal'] and setup['admitted_setup'] and setup['all_guards_match'] and not setup['working_overlay'] and not setup['overlays']
 assert trial['terminal'] and trial['all_guards_match'] and trial['ordinary_seconds']==10
 for group in ['source','images','tools','hosts']:
  assert trial[group+'_before']==trial[group+'_after']==setup[group+'_after']
  for name,h in trial[group+'_before'].items():
   key=label+'/source/'+name if group in ['source','images'] else absolute_record(name)
   assert rows[key]['sha256']==h,(label,group,name)
 assert trial['object_inputs']==trial['objects_after']
 for name,h in trial['object_inputs'].items():assert rows[absolute_record(name)]['sha256']==h
 admission=js(label+'/admission-result.json');assert admission['terminal'] and admission['accepted'] and not admission['other_jobs']
 return setup,trial
for label in ['comparison','tag-failure','borrowed-failure']:verify_selected(label)
for command in q['commands']:
 assert command['exit']==0 and not command.get('interrupted_or_timeout') and command['timeout'] in [60,600]
assert [v['label'] for v in q['commands'][:5]]==['strict-c-syntax','actual-c-reference-build','driver-nanoisa-build','driver-verify','driver-empty-argv']
assert q['commands'][2]['timeout']==600
expected_hashes={'original':'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0','scalar':'e710c83d560f1d3e60ff5d908fbd0260d64eef4fe71616446ad92c7764b427df'}
for group,count in [('original',1001),('scalar',25642)]:
 record=next(g for g in q['groups'] if g['name']==group);corpus=js('comparison/'+record['corpus_path'])
 assert record['cases']==corpus['cases']==len(corpus['corpus'])==count and rows['comparison/'+record['corpus_path']]['sha256']==record['corpus_sha256']
 expected=None
 for route in ['actual-c','nano-vm']:
  commands=[c for c in q['commands'] if c['label']==group+'-'+route];reports=[]
  for command in commands:
   assert command['timeout']==(600 if group=='original' else 60)
   prefix=1 if route=='actual-c' else 3;paths=command['argv'][prefix:]
   if group=='original':assert len(paths)<=8
   else:assert len(paths)<=32 and (len(paths)==1 or sum(rows[absolute_record(p)]['bytes'] for p in paths)<=8192)
   reports.extend(data('comparison/command-%04d.stdout'%command['number']).decode().splitlines())
  raw=('\n'.join(reports)+'\n').encode();assert raw==data('comparison/'+group+'-'+route+'.reports') and len(reports)==count and sha(raw)==expected_hashes[group]
  if expected is None:expected=reports
  else:assert reports==expected
  for case,text in zip(corpus['corpus'],reports):
   assert rows['comparison/'+case['path']]['sha256']==case['sha256']
   wanted=case['expected']
   if isinstance(wanted,str):
    if group=='original' and not case['name'].startswith(('global-algebra-','bounded-fifo-')):assert text.startswith(wanted)
    else:assert text==wanted
   else:
    valid,message,state=text.split('|',2);local,stack,globals_,context=state.split('/')
    assert valid==str(int(wanted['valid'])) and message==wanted['message'] and stack==wanted['stack'] and globals_=='9,' and context=='1,23,5,7,',case['name']
# My full caller-isolation assertions remain exact across all three driver versions.
driver=data('comparison/source/tests/nvm2c_affine_scalar_driver.nano')
assert driver==data('borrowed-failure/source/tests/nvm2c_affine_scalar_driver.nano')==data('tag-failure/source/tests/nvm2c_affine_scalar_driver.nano')
assert driver.count(b'assert (== (snapshot parsed.value) before)')==2
assert b'frame: (frame.clone result.frame)' in data('comparison/source/src_nano/nvm2c/affine_scalar.nano')
tag=js('tag-failure/light-result.json');assert tag['exception'] and tag['groups'][-1]['routes']==[]
assert b'literal-2-0-0-False' in data('tag-failure/exception.log')
red=js('tag-red/result.json');assert red['terminal'] and red['failure_reproduced'] and red['read_only_old_root'] and red['exit']==0 and red['timeout']==60
assert red['prior_receipt_sha256']==rows['tag-failure/light-result.json']['sha256']
assert [(v['actual_c_tag'],v['old_vm_tag']) for v in red['observations']]==[(3,2),(4,3),(2,4)]
assert red['guards_before']==red['guards_after']
for name,h in red['guards_before'].items():assert rows[absolute_record(name)]['sha256']==h
assert data('tag-red/stdout').decode().splitlines()==[v['old_vm_report'] for v in red['observations']]
old_c=data('tag-failure/scalar-actual-c.reports').decode().splitlines()
for row in red['observations']:assert old_c[row['index']]==row['actual_c_report']
borrowed=js('borrowed-failure/light-result.json');last=borrowed['commands'][-1]
assert len(borrowed['commands'])==1068 and last['label']=='scalar-nano-vm' and last['exit']==1 and last['timeout']==60 and not last.get('interrupted_or_timeout')
assert b'Assertion failed' in data('borrowed-failure/command-1068.stderr') and b'describe_scalar' in data('borrowed-failure/command-1068.stderr')
assert len(data('borrowed-failure/command-1068.stdout').decode().splitlines())==6
assert sha(data('borrowed-failure/scalar-actual-c.reports'))==expected_hashes['scalar']
cleanup_red=js('cleanup-red/result.json');green=js('cleanup-green/result.json')
assert cleanup_red['terminal'] and cleanup_red['old_test_helper_leak_reproduced'] and cleanup_red['new_scalar_control_pass']
assert data('cleanup-red/old-frame-zero-functions.stdout')==b'' and b'LeakSanitizer: detected memory leaks' in data('cleanup-red/old-frame-zero-functions.stderr')
assert green['terminal'] and green['all_controls_pass'] and green['baseline_receipt_sha256']==rows['cleanup-red/result.json']['sha256'] and green['source_guards']==green['source_after']
for name,h in green['source_guards'].items():assert rows[absolute_record(name)]['sha256']==h
for command in green['commands']:assert command['exit']==0 and command['timeout']==60 and not command.get('timed_out')
for name in ['old-zero-functions','new-zero-functions']:
 assert data('cleanup-green/'+name+'.bin')==data('cleanup-red/'+name+'.bin')
 assert data('cleanup-green/'+name+'.stdout')==b'D\n' and data('cleanup-green/'+name+'.stderr')==b''
assert '-fsanitize=address,undefined' in green['commands'][0]['argv'] and '-fno-sanitize-recover=all' in green['commands'][0]['argv']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git(pin,name):
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
 raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for label in ['comparison','tag-failure','borrowed-failure']:
 setup=js(label+'/result.json')
 for name,h in setup['source_before'].items():assert sha(git(setup['source_base'],name))==h,(label,name)
assert data('cleanup-green/source/tests/nvm2c_affine_frame_reference.c')==git(m['test_cleanup_source_commit'],'tests/nvm2c_affine_frame_reference.c')
for name,h in s['source_before'].items():
 if name.startswith('src/') and name.endswith(('.c','.h','.inc')):assert sha(git('95a4062efebe0e97d8cde271117eac1deab21595',name))==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: exact9d scalar/frame C-VM comparison, all3837 commands, original1001/new25642 complete reports, unchanged assertions/bounds, preserved tag and borrowed-failure roots, actual red literals and same-input sanitizer cleanup. No full-five/paired/CFG/module/cutover/release acceptance.'%len(rows))
