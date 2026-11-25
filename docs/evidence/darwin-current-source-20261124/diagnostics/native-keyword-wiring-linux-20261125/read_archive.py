"""I verify keyword storage witnesses and preserve distinct regression failures."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'75e4bf3c592373245482ae696426a2f208fbc5311632d2cdd8730689c28c05aa');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('qualification/setup-result.json');f=js('qualification/focused-result.json');e=js('qualification/extra-tests/result.json');r=js('qualification/red-green/result.json');g=js('qualification/regressions-result.json');b=js('qualification/baseline-regressions-result.json');v=js('complete-inputs/result.json');l=js('complete-inputs/legacy-result.json')
assert m['component_only'] and not any(m[k] for k in ['whole_release','new_bootstrap','paired','complete_regressions'])
assert s['terminal'] and s['admitted_setup'] and s['cold_tools'] and s['working_overlay'] and not s['provider_copies'] and s['ordinary_shadow_seconds']==10
assert s['source_before']==s['source_after'] and s['filesystem']!='nfs' and not s['other_nanolang_jobs'] and all(x['cpu_seconds']>=1 and x['share']>=.75 for x in s['cpu_rounds'])
for name,h in s['source_after'].items():assert rows['qualification/source/'+name]['sha256']==h
for name,h in s['qualified_images'].items():assert rows['qualification/source/'+name]['sha256']==h
for name,h in s['tools'].items():assert rows['external/'+name.lstrip('/')]['sha256']==h
assert f['terminal'] and f['passed'] and f['exit']==0 and f['all_guards_match'] and b'Ran 3 tests' in data('qualification/focused.stderr') and b'OK' in data('qualification/focused.stderr')
assert e['terminal'] and e['passed'] and e['tests']==1 and e['failures']==e['errors']==0 and e['source_guards_match'] and e['image_guards_match']
assert r['terminal'] and r['passed'] and r['source_guards_match'] and r['image_guards_match'] and r['tools_guards_match']
commands={x['label']:x for x in r['commands']};assert len(commands)==13
for name in ['control','local','parameter','shadow']:
 assert r['sources'][name]==rows['qualification/red-green/'+name+'.nano']['sha256']==rows['original-minimal/'+name+'.nano']['sha256']
 for axis in ['old','new']:
  assert commands[axis+'-'+name]['exit']==(1 if axis=='old' and name!='control' else 0)
  if axis=='new' or name=='control':
   assert commands[axis+'-'+name+'-execute']['exit']==0
   assert r['products'][axis+'-'+name]['sha256']==rows['qualification/red-green/'+axis+'-'+name]['sha256']
  else:assert data('qualification/red-green/'+axis+'-'+name)==b'prior-value'
assert g['terminal'] and g['tests_run']==65 and g['failures']==4 and g['errors']==3 and not g['passed'] and g['all_guards_match'] and g['skips']==0
assert b['terminal'] and b['tests_run']==3 and b['failures']==3 and b['errors']==b['skips']==0
assert v['terminal'] and v['passed'] and v['tests']==23 and v['failures']==v['errors']==v['skips']==0
assert l['terminal'] and not l['passed'] and l['tests']==23 and l['errors']==l['skips']==0
for q in [v,l]:assert all(q[k] for k in ['source_guards_match','image_guards_match','fixture_guards_match','copied_images_match'])
for name,h in v['fixtures'].items():assert rows['complete-inputs/fixture-view/'+name]['sha256']==h
for name,h in v['images'].items():assert rows['complete-inputs/fixture-view/bin/'+name]['sha256']==rows['qualification/source/bin/'+name]['sha256']==h
for name,q in g['retained_native_stages'].items():assert rows['baseline/bin/'+name]['sha256']==q['sha256']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def blob(pin,name):
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in s['source_before'].items():
 if name not in s['overlays']:assert sha(blob(s['source_pin'],name))==h,name
for name in ['src/transpiler.c','src/transpiler_iterative_v3_twopass.c','Makefile.gnu','tests/nanoisa/fixtures/native_keyword_bindings.nano']:assert sha(blob(m['source_commit'],name))==s['overlays'][name]
assert sha(blob(m['source_commit'],'tests/test_native_keyword_bindings.py'))==rows['qualification/extra-tests/test_native_keyword_bindings.py']['sha256']
for name,h in v['fixtures'].items():assert sha(blob(s['source_pin'],name))==h
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: fresh strict C tools, focused3 plus separate effect control,13 red/green commands, default23 PASS; failed65 and baseline3 and full legacy23 retained. No complete-regression, bootstrap, paired, cutover or release acceptance.'%len(rows))
