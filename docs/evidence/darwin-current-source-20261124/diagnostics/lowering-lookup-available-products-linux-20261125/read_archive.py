"""I verify available products while retaining the complete gate's failed boundary."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'490845fcb0df21edee17b2310fc2b7904a5a26b2f972cb4f9d69c8d2259b215e');sha=lambda raw:hashlib.sha256(raw).hexdigest();q=js('diagnostic/result.json');origin=js('origin/cohort-result.json')
assert m['diagnostic_only'] and m['reused_products'] and not any(m[k] for k in ['new_builds','full_five_routes','whole_release'])
assert q['terminal'] and q['diagnostic_only'] and q['reused_products'] and not q['new_builds'] and not q['full_five_routes'] and q['source_producer_timeout_unrepaired'] and q['passed'] and q['tests']==1 and q['failures']==q['errors']==q['skips']==0
assert all(q[k] for k in ['source_guards_match','image_guards_match','product_guards_match','origin_cohort_unchanged']) and q['origin_cohort_sha256']==rows['origin/cohort-result.json']['sha256']
assert origin['terminal'] and origin['all_guards_match'] and origin['commands'][-1]['timed_out'] and origin['commands'][-1]['timeout']==600 and origin['commands'][-1]['exit']==-9 and 'permanent_test' not in origin
assert q['permanent_test_sha256']==rows['source/tests/test_nanoisa_lookup_projection.py']['sha256']
expected=js('source/tests/nanoisa_lookup_projection_cases.json');cases=js('source/tests/typecheck_scope_copy_cases.json')['cases'];assert len(cases)==len(expected)==24 and len(q['commands'])==75
for index,command in enumerate(q['commands']):
 assert command['number']==index+1 and command['exit']==0 and command['timeout']==60 and not command.get('timed_out')
 value=data('diagnostic/command-%d.stdout'%(index+1))
 if index<3:assert value==b'LOOKUP-PROJECTION-OK\n'
 else:assert sha(value)==expected[(index-3)//3]['report_sha256']
for i,(case,wanted) in enumerate(zip(cases,expected)):
 assert sha(bytes.fromhex(case['source_hex']))==wanted['source_sha256']==rows['diagnostic/inputs/case-%d.nano'%i]['sha256']
origin_root=Path(js('origin/result.json')['producer']).parents[2]
for path,h in q['product_hashes'].items():
 name=Path(path);prefix=origin_root;assert rows['reused/'+str(name.relative_to(prefix))]['sha256']==h
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
for name in ['tests/test_nanoisa_lookup_projection.py','tests/nanoisa_lookup_projection_cases.json','tests/typecheck_scope_copy_cases.json']:assert subprocess.check_output(['git','show',m['source_commit']+':'+name],cwd=repo)==data('source/'+name)
print('PASS %d records: unchanged permanent method/all24 inputs,75 available C-native/VM/instrumented-native runs, no rebuild; original source-produced600 failure unchanged. No full-five, new source producer, factory, bootstrap, paired, cutover or release acceptance.'%len(rows))
