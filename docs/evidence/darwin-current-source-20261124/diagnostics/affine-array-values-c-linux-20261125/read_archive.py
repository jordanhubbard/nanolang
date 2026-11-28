"""I verify new positive element contracts against the unchanged qualified C image."""
from pathlib import Path
import ast,hashlib,subprocess,sys
from itertools import product
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'839110e96ed16689bcac613fc6a412ee7cfc7f57b26fb6013489b5d790322cac')
r=js('comparison/result.json');sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert m['c_oracle_companion_only'] and not m['full_five_routes'] and not m['whole_release']
assert r['terminal'] and r['accepted_c_companion'] and r['all_guards_match'] and not r['failures']
assert m['source_commit']==r['source_pin']=='1ce3e8ec50ebafd9b0cc61ffd331f02cb46b58f5'
cm,cr,cd,cj=read_archive((base/m['core_manifest']['path']).parent,m['core_manifest']['sha256'])
assert m['core_manifest']['sha256']==r['prior_archive_manifest_sha256']=='f3d257f49f4fcd6c149009800a71ea65a33b9131e0f8b7c25fa60240cc0b9234'
core=cj('comparison/result.json');assert core['terminal'] and core['accepted_c_oracle'] and core['all_guards_match']
assert sha(cd('comparison/result.json'))==r['prior_result_sha256']
assert r['reference_sha256']==r['reference_after']==rows['selected-reference']['sha256']==cr['comparison/products/reference']['sha256']
assert r['source_before']==r['source_after']
for name,h in r['source_before'].items():assert rows['source/'+name]['sha256']==h
for group in ['objects','tools','production_source']:
 assert r[group+'_before']==r[group+'_after']==core[group+'_after']
 for n,h in r[group+'_before'].items():assert rows['external/'+n.lstrip('/')]['sha256']==h
assert r['unchanged_core_count']==16840 and r['unchanged_core_report_sha256']==sha(cd('comparison/arrays.reports'))=='52b6050f5e7943a152142bd8e6bb326da4ef65858d2ad14b947edc27a9575961'
assert len(r['cases'])==216 and len(r['commands'])==28
expected_names={f'positive-{name}-{resource}-{op}' for name,resource,op in product(['int','u8','float','bool','string','record','union','array'],[False,True],range(80,89))}
assert expected_names=={c['name'] for c in r['cases'] if c['name'].startswith('positive-')}
observed=[]
for i,c in enumerate(r['commands'],1):
 assert c['argv'][0]==r['reference'] and c['exit']==0 and c['timeout']==60 and data('comparison/command-%04d.stderr'%i)==b''
 paths=c['argv'][1:];assert len(paths)<=32
 sizes=[rows['comparison/'+str(Path(p).relative_to(m['roots']['comparison']))]['bytes'] for p in paths]
 assert len(paths)==1 or sum(sizes)<=8192
 observed+=data('comparison/command-%04d.stdout'%i).decode().splitlines()
assert len(observed)==216
raw=('\n'.join(observed)+'\n').encode();assert raw==data('comparison/values.reports') and sha(raw)==r['report_sha256']
assert sha(cd('comparison/arrays.reports')+raw)==r['combined_report_sha256']
for c,text in zip(r['cases'],observed):
 path='comparison/'+str(Path(c['path']).relative_to(m['roots']['comparison']));assert rows[path]['sha256']==c['sha256']
 e=c['expected'];valid,message,state=text.split('|',2);local,stack,relations,context=state.split('/')
 assert valid==str(int(e['valid'])) and message==e['message'] and stack==e['stack'] and relations=='9,' and context=='1,23,5,7,',c['name']
# I verify the previous generator body and C implementation did not change.
old=ast.parse(cd('comparison/tests/test_nanolang_translator_affine_arrays.py'));new=ast.parse(data('source/tests/test_nanolang_translator_affine_arrays.py'))
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
assert [ast.dump(n) for n in function(old,'array_cases').body]==[ast.dump(n) for n in function(new,'array_core_cases').body]
assert ast.dump(function(old,'constructor_model'))==ast.dump(function(new,'constructor_model'))
changed=subprocess.check_output(['git','diff','--name-only','5f3f2c360d',r['source_pin'],'--',':(top)src/',':(top)src_nano/',':(top)tests/',':(top)Makefile.gnu'],cwd=base,text=True).splitlines()
assert changed==['tests/test_nanolang_translator_affine_arrays.py']
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for n,h in r['source_before'].items():
 proc.stdin.write((r['source_pin']+':'+n+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';blob=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(blob)==h,n
proc.stdin.close();assert proc.wait()==0
print('PASS %d new/%d core records; all216 new element cases/all28 commands, unchanged qualified C image, exact guards, preserved16840 core generator and all17056 combined report bytes. The core execution is retained separately; no new full rerun, NanoLang/sanitizer/paired/CFG/cutover/release acceptance.'%(len(rows),len(cr)))
