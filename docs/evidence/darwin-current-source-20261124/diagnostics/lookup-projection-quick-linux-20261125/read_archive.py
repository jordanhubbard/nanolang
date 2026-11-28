"""I verify the source projection, retained products and complete baseline diagnostics."""
from pathlib import Path
import sys,json,hashlib,subprocess,re
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'8daef2c428d6701398da9829b3d5fb016cc6b3e2e0c8ef9676e050a5ed4b59b0');sha=lambda b:hashlib.sha256(b).hexdigest();s=js('quick/result.json');a=js('quick/admission-result.json');q=js('quick/quick-result.json');ob=js('quick/materialization.json');pre=js('preflight.json')
assert m['source_commit']=='8806139ca8bf3f60a25d88d241f67747c6ef9e8a' and m['baseline_source_commit']==s['source_base']==q['source_base']==pre['source_base']=='189578c89cf1c7b84e3e83f0ccde72e30c784045'
assert m['component_only'] and m['c_native_vm_only'] and m['materialization_observation'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','new_compiler_image','whole_release','historical_cause','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and s['working_overlay'] and s['overlays']==m['overlays']==pre['overlays']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3 and all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert q['terminal'] and q['accepted_c_native_vm_boundary'] and q['all_guards_match'] and q['ordinary_seconds']==10 and not q['full_five_routes']
assert len(q['commands'])==108 and len(q['cases'])==24
for c in q['commands']:assert c['exit']==0 and not c.get('timed_out') and c['timeout']==(600 if c['number'] in [1,3,6,7,9,12] else 60)
for n in [2,5,8,11]:assert data('quick/quick-%03d.stdout'%n)==b'SCOPE-COPY-OK\n'
corpus=js('quick/source/tests/typecheck_scope_copy_cases.json')['cases'];assert len(corpus)==24
for i,case in enumerate(corpus):
 row=q['cases'][i];source=bytes.fromhex(case['source_hex']);expected=case['expected'].encode();assert row['source_sha256']==sha(source) and row['expected_sha256']==sha(expected) and row['commands']==list(range(13+i*4,17+i*4));assert data('quick/diagnostics/case-%02d.nano'%i)==source
 for n in row['commands']:assert data('quick/quick-%03d.stdout'%n)==expected
for g in ['source','images','tools','hosts']:
 assert q[g+'_before']==q[g+'_after']==s[g+'_before']==s[g+'_after']
 for name,h in q[g+'_before'].items():assert rows['quick/source/'+name if g in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['inputs_before']==q['inputs_after']
for name,h in q['inputs_before'].items():assert rows['quick/source/'+name]['sha256']==h
for name,v in q['products'].items():assert rows['quick/'+name]['sha256']==v['sha256'] and rows['quick/'+name]['bytes']==v['bytes']
assert ob['terminal'] and ob['actual_retained_translations'] and ob['accepted_materialization_observation'] and not ob['new_build'] and not ob['performance_claim']
for label in ['baseline','candidate']:
 facts=ob['functions'][label];raw=data('quick/products/'+label+'.c');text=raw.decode();matches=list(re.finditer(r'^static int64_t (\w*_env_lookup)\([^\n]*\) \{$',text,re.M));assert len(matches)==1;start=matches[0].start();body=text[start:text.index('\n}\n',start)+3];assert body.encode()==data('quick/'+label+'-env_lookup.c') and facts['function_sha256']==sha(body.encode()) and facts['product_sha256']==sha(raw)
 locals_match=re.search(r'nrec_t \*rl = calloc\((\d+), sizeof \*rl\);',body);temps_match=re.search(r'nrec_t \*r = (\d+) \? calloc',body);assert temps_match
 assert facts['record_locals']==(int(locals_match.group(1)) if locals_match else 0)==(1 if label=='baseline' else 0)
 assert facts['record_temporaries']==int(temps_match.group(1))==(2 if label=='baseline' else 1)
 assert facts['record_local_roots']==len(re.findall(r'nroot_add\(&nroots.live, 4, &rl\[',body))==(1 if label=='baseline' else 0)
 assert facts['array_record_gets']==body.count('nrarr_get(')==1 and facts['empty_print_calls']==body.count('fputs(')==1 and facts['flush_calls']==body.count('fflush(')==1
assert all(pre[k] for k in ['only_production_name_projection_changed','empty_print_miss_branch_bytes_unchanged','reverse_order_and_signature_unchanged','all_old_assertions_preserved','all_old_fixture_bytes_outside_added_controls_unchanged','all14_sizes_and24_complete_diagnostics_unchanged']) and pre['old_typechecker_assert_lines']==833
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git_data(commit,name):
 p.stdin.write((commit+':'+name+'\n').encode());p.stdin.flush();head=p.stdout.readline().split();assert len(head)==3 and head[1]==b'blob',name;raw=p.stdout.read(int(head[2]));assert p.stdout.read(1)==b'\n';return raw
for name,h in q['source_before'].items():assert sha(git_data(m['source_commit'],name))==h,name
for name,h in pre['overlays'].items():assert sha(data('quick/source/'+name))==h
old=git_data(m['baseline_source_commit'],'src_nano/typecheck.nano');new=data('quick/source/src_nano/typecheck.nano');function=lambda raw:raw.split(b'fn env_lookup(',1)[1].split(b'\nfn last_index_of_dot(',1)[0]
assert function(new).replace(b'        if (== (at symbols i).name name) {',b'        let sym: Symbol = (at symbols i)\n        if (== sym.name name) {',1)==function(old)
assert data('quick/source/src_nano/typecheck_baseline.nano')==old
fixture=data('quick/source/tests/typecheck_scope_copy.nano');assert data('quick/source/tests/typecheck_scope_copy_baseline.nano')==fixture.replace(b'import "src_nano/typecheck.nano"',b'import "src_nano/typecheck_baseline.nano"',1)
addition=fixture[fixture.index(b'fn scope_lookup_names()'):fixture.index(b'fn scope_report_octets(')];assert fixture.replace(addition,b'',1).replace(b'fn main() -> int {\n    (scope_lookup_names)',b'fn main() -> int {',1).replace(b'(scope_function_tables) (scope_lookup_names) }',b'(scope_function_tables) }',1)==git_data(m['baseline_source_commit'],'tests/typecheck_scope_copy.nano')
assert git_data(m['baseline_source_commit'],'tests/typecheck_scope_copy_cases.json')==data('quick/source/tests/typecheck_scope_copy_cases.json')
asserts=lambda raw:[x.strip() for x in raw.splitlines() if x.strip().startswith(b'assert ')];it=iter(asserts(new));assert len(asserts(old))==833 and all(any(x==v for x in it) for v in asserts(old))
p.stdin.close();assert p.wait()==0
print('PASS %d records: actual baseline/candidate C-native+VM controls,108 commands/all24 complete original reports per route and selected guard maps; exact source projection and retained C local1->0/temporaries2->1/root1->0, original print/flush behavior. No full-five-route sanitizer, current compiler/factory/bootstrap/paired/full-translator/release or performance acceptance.'%len(rows))
