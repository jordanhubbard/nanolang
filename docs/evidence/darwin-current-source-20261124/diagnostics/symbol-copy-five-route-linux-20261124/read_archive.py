"""I verify the full copy component and distinguish retained-product assertions."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'7ea465b1d257bc916b60b67ed34d66532afda538fd39957f8fb16a43e0dcab04');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('component/result.json');q=js('component/cohort-result.json');a=js('component/admission-result.json');r=js('regression/result.json');isolation=js('component/fixture-isolation.json');preflight=js('regression-source/preflight.json')
assert m['source_commit']==s['source_base']=='e2681188ec5213d160e92925b70878418b5e64ba' and m['overlay_source_commit']=='540292feacdb716aba087fbe4f03d71550e211ec'
assert m['diagnostic_only'] and not any(m[k] for k in ['product_acceptance','historical_cause','new_bootstrap','whole_release','literal_bug_fixed','make_gate_run']) and m['regression_products_reused']
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and s['working_overlay'] and not s['stage_alias']
assert a['accepted'] and a['terminal'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3
assert all(c['cpu']>=1 and c['share']>=.75 for c in s['cpu_rounds'])
assert q['terminal'] and q['accepted_component'] and q['all24_five_routes_equal_original'] and q['ordinary_seconds']==10 and not q['literal_bug_fixed']
assert len(q['routes'])==5 and len(q['commands'])==132 and len(q['cases'])==24
assert [row['label'] for row in q['routes']]==['c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native']
for row in q['commands']:assert row['exit']==0 and not row.get('timed_out') and row['timeout']==(600 if row['number']<=7 else 60)
for number in [4,7]:
 command=q['commands'][number-1]['argv'];assert all(flag in command for flag in ['-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])
for number in range(8,13):assert data('component/command-%d.stdout'%number)==b'SCOPE-COPY-OK\n'
original=base.parent/'symbol-copy-record-literal-boundary-linux-20261124';om,orr,od,oj=read_archive(original,'b1e89cf7e662a215661a6bfc50fc4a692acfa6106cd3973710504fe34391001a')
original_cases=oj('component/diagnostics/cases.json')['cases'];current_cases=js('component/diagnostics/cases.json')['cases'];assert current_cases==original_cases
for index,case in enumerate(q['cases']):
 expected=data('component/diagnostics/case-%02d.c-native-expected'%index);assert expected==od('component/diagnostics/case-%02d.c-native-expected'%index)
 assert case['report_sha256']==sha(expected) and case['source_sha256']==rows['component/diagnostics/'+current_cases[index]['path']]['sha256']==orr['component/diagnostics/'+current_cases[index]['path']]['sha256']
 assert case['command_numbers']==list(range(13+index*5,18+index*5))
 for number in case['command_numbers']:assert data('component/command-%d.stdout'%number)==expected
for group in ['source','images','tools','hosts']:
 assert q[group+'_before']==q[group+'_after']==s[group+'_before']==s[group+'_after']
 for name,h in q[group+'_before'].items():assert rows[('component/source/'+name) if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['all_guards_match'] and q['inputs_before']==q['inputs_after']
for name,h in q['inputs_before'].items():assert rows['component/diagnostics/'+name]['sha256']==h
assert q['source_producer_sha256']==q['producer_after']==s['producer_sha256']==rows['component/source/bin/nanoc_emitter_phase']['sha256']
assert q['commands'][4]['argv'][0]==s['producer'] and q['commands'][4]['argv'][-3]=='--emit-nvm'
for name,v in q['products'].items():assert rows['component/products/'+name]['sha256']==v['sha256'] and rows['component/products/'+name]['bytes']==v['bytes']
assert q['products']['decoder.nvm']['sha256']=='c6abe1e3e87a953c3942132748a8bfb46e79bcbf28b273cb4e87909905188241' and q['products']['decoder.nvm']['bytes']==790760
assert q['products']['source-decoder.nvm']['sha256']=='c00c191df94527046f99b1d4b9b2535b3be04da1be9f0698308066f7fe695a50' and q['products']['source-decoder.nvm']['bytes']==462320
assert isolation['all_assert_lines_unchanged'] and isolation['all14_copy_sizes_unchanged'] and isolation['production_env_copy_body_unchanged'] and not isolation['literal_bug_fixed']
assert isolation['original_tc_assert_lines']==825 and isolation['original_fixture_assert_lines']==28
assert r['terminal'] and r['accepted_assertion_layer'] and r['all_guards_match'] and r['assertion_layer_only'] and r['uses_retained_real_products'] and not r['new_build'] and r['make_gate_not_run']
assert r['tests_run']==r['discovered']==2 and not any(r[k] for k in ['failures','errors','skips'])
assert r['products_before']==r['products_after']==q['products'] and r['origin_receipt_sha256']==rows['component/cohort-result.json']['sha256']
assert len(r['builder_requests'])==1 and not r['builder_requests'][0]['generation_repeated'] and r['builder_requests'][0]['fixture_sha256']==rows['component/source/tests/typecheck_scope_copy.nano']['sha256']
assert b'Ran 2 tests' in data('regression/unittest.log') and data('regression/unittest.log').rstrip().endswith(b'OK')
assert preflight['make_append_only'] and preflight['old_dependencies_modules_limits_unchanged'] and preflight['all24_original_inputs_reports_unchanged'] and preflight['runtime_not_run'] and len(preflight['two_methods'])==2
assert r['integration_inputs_before']==r['integration_inputs_after']
for name,h in r['integration_inputs_before'].items():assert rows['regression-source/'+name]['sha256']==h
corpus=js('regression-source/tests/typecheck_scope_copy_cases.json');assert len(corpus['cases'])==24 and corpus['literal_bug']==1256
for index,case in enumerate(corpus['cases']):
 assert bytes.fromhex(case['source_hex'])==data('component/diagnostics/'+current_cases[index]['path']) and case['expected'].encode()==data('component/diagnostics/case-%02d.c-native-expected'%index)
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git_data(commit,name):
 proc.stdin.write((commit+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',name;raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in s['source_before'].items():assert sha(git_data(m['overlay_source_commit'] if name in m['overlay_files'] else m['source_commit'],name))==h,name
for name,h in r['integration_inputs_before'].items():assert sha(git_data(m['harness_source_commit'],name))==h,name
old_tc=git_data('6e578585bb2eb6cbffc26db63bbaa1f19ec4dede','src_nano/typecheck.nano');new_tc=data('component/source/src_nano/typecheck.nano');old_fixture=git_data('6e578585bb2eb6cbffc26db63bbaa1f19ec4dede','tests/typecheck_scope_copy.nano');new_fixture=data('component/source/tests/typecheck_scope_copy.nano')
asserts=lambda raw:[line.strip() for line in raw.splitlines() if line.strip().startswith(b'assert ')]
assert asserts(old_tc)==asserts(new_tc) and asserts(old_fixture)==asserts(new_fixture)
function=lambda raw:raw.split(b'fn env_copy_symbols(',1)[1].split(b'\nshadow env_copy_symbols {',1)[0]
assert function(old_tc)==function(new_tc)
assert [line for line in old_fixture.splitlines() if b'for count in' in line]==[line for line in new_fixture.splitlines() if b'for count in' in line]
old_make=git_data(m['overlay_source_commit'],'Makefile.gnu');new_make=data('regression-source/Makefile.gnu');block=b'\n.PHONY: test-typecheck-scope-copy\ntest-typecheck-scope-copy: $(COMPILER_C) nano_virt nano_vm nvm2c nvm2c-runtime\n\tpython3 -m unittest -v tests.test_typecheck_scope_copy\n\ntest-units: test-typecheck-scope-copy\n';assert new_make.count(block)==1 and new_make.replace(block,b'',1)==old_make
proc.stdin.close();assert proc.wait()==0
assert s['producer_sha256']==oj('component/result.json')['producer_sha256']
print('PASS %d records: fresh stock five-route build,14 copy sizes and24 complete original diagnostic records per route; both generated ASan/UBSan natives pass; unchanged825/28 assertion lines and production copy body, original literal failure retained. New two-method assertion layer reuses exact guarded products; Make not run. No new compiler/factory/Stage2/paired/pure-translator/release or literal-bug acceptance.'%len(rows))
