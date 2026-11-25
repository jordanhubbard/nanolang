"""I verify bodyless scope isolation, full routes and the retained observer failure."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'3573bb1d279a529de31bf9b65f0790aeda9aa5bb156fdcdc947ac83666e5a146');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('component/result.json');q=js('component/cohort-result.json');a=js('component/admission-result.json');first=js('quick/quick-result.json');c=js('comparison/result.json');pre=js('preflight.json')
assert m['source_commit']==s['source_base']=='ee78adba9c7754e36761a3cda76b1218840198b3'
assert m['baseline_source_commit']==pre['source_base']=='0a26c4654f7e40d623af41c1591a476d0eccb74a'
assert m['component_only'] and m['full_five_routes_pass'] and m['retained_comparison_products'] and m['first_comparison_protocol_failure']
assert not any(m[k] for k in ['literal_bug_fixed','new_compiler_image','new_bootstrap','whole_release','historical_cause','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['stage_alias']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3
assert all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert q['terminal'] and q['accepted_component'] and q['all_guards_match'] and q['all24_five_routes_equal_original'] and q['ordinary_seconds']==10
assert len(q['commands'])==132 and len(q['cases'])==24
assert [v['label'] for v in q['routes']]==['c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native']
for command in q['commands']:assert command['exit']==0 and not command.get('timed_out') and command['timeout']==(600 if command['number']<=7 else 60)
for n in [4,7]:assert all(flag in q['commands'][n-1]['argv'] for flag in ['-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])
for n in range(8,13):assert data('component/command-%d.stdout'%n)==b'SCOPE-COPY-OK\n'
corpus=js('component/source/tests/typecheck_scope_copy_cases.json')['cases'];assert len(corpus)==24
for i,case in enumerate(corpus):
 expected=case['expected'].encode();source=bytes.fromhex(case['source_hex']);rec=q['cases'][i]
 assert rec['source_sha256']==sha(source) and rec['report_sha256']==sha(expected) and rec['command_numbers']==list(range(13+i*5,18+i*5))
 assert data('component/diagnostics/case-%02d.nano'%i)==source and data('component/diagnostics/case-%02d.c-native-expected'%i)==expected
 for n in rec['command_numbers']:assert data('component/command-%d.stdout'%n)==expected
for group in ['source','images','tools','hosts']:
 assert q[group+'_before']==q[group+'_after']==s[group+'_before']==s[group+'_after']
 for name,h in q[group+'_before'].items():assert rows[('component/source/'+name) if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['inputs_before']==q['inputs_after']
for name,h in q['inputs_before'].items():assert rows['component/diagnostics/'+name]['sha256']==h
assert q['source_producer_sha256']==q['producer_after']==s['producer_sha256']==rows['component/source/bin/nanoc_emitter_phase']['sha256']=='2eb02fd33d8cc8d8ca5f2684b4399d55540b648a1f4eb2570e6ead118e03a63d'
for name,v in q['products'].items():assert rows['component/products/'+name]['sha256']==v['sha256'] and rows['component/products/'+name]['bytes']==v['bytes']
assert first['terminal'] and first['exception']=='AssertionError()' and len(first['commands'])==11 and all(v['exit']==0 for v in first['commands'])
assert data('quick/quick-011.stdout')==corpus[0]['expected'].encode()
assert b"expected=b'SCOPE-COPY-OK\\n'+case['expected'].encode()" in data('quick/quick.py')
assert c['terminal'] and c['accepted_c_native_vm_diagnostics'] and c['all_guards_match'] and c['retained_products'] and not c['new_build'] and not c['full_five_routes']
assert c['origin_receipt_sha256']==c['origin_receipt_after']==rows['quick/quick-result.json']['sha256'] and len(c['commands'])==98 and len(c['cases'])==24
for group in ['source','images','inputs','tools','hosts','module_hosts','products']:assert c[group+'_before']==c[group+'_after']
for command in c['commands']:assert command['exit']==0 and command['timeout']==60 and not command.get('timed_out')
for i,rec in enumerate(c['cases']):
 assert rec['source_sha256']==sha(bytes.fromhex(corpus[i]['source_hex'])) and rec['expected_sha256']==sha(corpus[i]['expected'].encode())
 assert rec['commands']==list(range(3+i*4,7+i*4))
 for n in rec['commands']:assert data('comparison/command-%03d.stdout'%n)==corpus[i]['expected'].encode()
for name,v in c['products_before'].items():assert rows['quick/products/'+Path(name).name]['sha256']==v['sha256']
for name,h in c['module_hosts_before'].items():assert rows['external/'+name.lstrip('/')]['sha256']==h
assert all(pre[k] for k in ['only_production_clone_and_append_guards_changed','parameter_and_return_validation_order_and_defined_body_bytes_unchanged','all_old_assertions_preserved','all_old_fixture_bytes_outside_added_controls_unchanged','all14_sizes_and24_complete_diagnostics_unchanged'])
assert pre['original_typechecker_assert_lines']==825
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git_data(commit,name):
 proc.stdin.write((commit+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',name;raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in s['source_before'].items():assert sha(git_data(m['source_commit'],name))==h,name
for name,h in c['source_before'].items():assert sha(git_data(m['source_commit'] if name in pre['overlays'] else m['baseline_source_commit'],name))==h,name
for name,h in pre['overlays'].items():assert sha(data('component/source/'+name))==h
assert data('quick/source/src_nano/typecheck_baseline.nano')==git_data(m['baseline_source_commit'],'src_nano/typecheck.nano')
fixture=data('component/source/tests/typecheck_scope_copy.nano');assert data('quick/source/tests/typecheck_scope_copy_baseline.nano')==fixture.replace(b'src_nano/typecheck.nano',b'src_nano/typecheck_baseline.nano',1)
old=git_data(m['baseline_source_commit'],'src_nano/typecheck.nano');new=data('component/source/src_nano/typecheck.nano')
function=lambda raw:raw.split(b'fn check_function(',1)[1].split(b'\nshadow check_function {',1)[0]
restored=function(new).replace(b'    # I validate every parameter below. Only a body needs a private lexical table.\n    let mut func_symbols: array<Symbol> = symbols\n    if (>= func.body 0) { set func_symbols (env_copy_symbols symbols) }',b'    /* Create a fresh symbol table for this function.\n     * All symbols at this point are functions or mutable globals (set up in Phase 1),\n     * so no filtering is needed \xe2\x80\x94 just use them all. This avoids the O(n^2) rebuild. */\n    let mut func_symbols: array<Symbol> = (env_copy_symbols symbols)').replace(b'        if (>= func.body 0) { set func_symbols (env_add_symbol func_symbols psym) }',b'        set func_symbols (env_add_symbol func_symbols psym)')
assert restored==function(old)
old_assert=[v.strip() for v in old.splitlines() if v.strip().startswith(b'assert ')];new_assert=[v.strip() for v in new.splitlines() if v.strip().startswith(b'assert ')]
it=iter(new_assert);assert all(any(x==v for x in it) for v in old_assert) and len(old_assert)==825
assert git_data(m['baseline_source_commit'],'tests/typecheck_scope_copy_cases.json')==data('component/source/tests/typecheck_scope_copy_cases.json')
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: bodyless scope controls, 14 copy sizes and 24 complete original reports across five routes; both generated sanitizer natives pass. Original observer prefix failure retained; corrected 98-command comparison reuses exact baseline/candidate products. No new compiler image, bootstrap, factory, paired platform, full translator or release acceptance.'%len(rows))
