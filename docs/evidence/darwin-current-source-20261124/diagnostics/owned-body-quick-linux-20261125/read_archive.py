"""I verify the owned-function scope boundary and preserve its first fixture refusal."""
from pathlib import Path
import sys,json,hashlib,re,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'2459c0248166402ac204ae87db61a75751403a3e4a2f00c2a619b138fa95e7ad');sha=lambda b:hashlib.sha256(b).hexdigest();s=js('quick/result.json');a=js('quick/admission-result.json');q=js('quick/quick-result.json');pre=js('preflight.json');ob=js('quick/call-boundaries.json')
assert m['source_commit']=='00025ff76979b4009b84f10f8cb8f07492c62210' and m['baseline_source_commit']==s['source_base']==q['source_base']==pre['source_base']=='ebe6386487a9b22075c1e607c1ca6d23794237d3'
assert m['component_only'] and m['c_native_vm_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','new_compiler_image','whole_release','historical_cause','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and s['working_overlay'] and s['overlays']==m['overlays']==pre['changes']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3 and all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert q['terminal'] and q['accepted_c_native_vm_boundary'] and q['all_guards_match'] and q['ordinary_seconds']==10 and not q['full_five_routes']
assert len(q['commands'])==108 and len(q['cases'])==24
for c in q['commands']:assert c['exit']==0 and not c.get('timed_out') and c['timeout']==(600 if c['number'] in [1,3,6,7,9,12] else 60)
for n in [2,5,8,11]:assert data('quick/quick-%03d.stdout'%n)==b'SCOPE-COPY-OK\n'
corpus=js('quick/source/tests/typecheck_scope_copy_cases.json')['cases'];assert len(corpus)==24
for i,case in enumerate(corpus):
 r=q['cases'][i];source=bytes.fromhex(case['source_hex']);expected=case['expected'].encode();assert r['source_sha256']==sha(source) and r['expected_sha256']==sha(expected) and r['commands']==list(range(13+i*4,17+i*4));assert data('quick/diagnostics/case-%02d.nano'%i)==source
 for n in r['commands']:assert data('quick/quick-%03d.stdout'%n)==expected
for g in ['source','images','tools','hosts']:
 assert q[g+'_before']==q[g+'_after']==s[g+'_before']==s[g+'_after']
 for name,h in q[g+'_before'].items():assert rows['quick/source/'+name if g in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
assert q['inputs_before']==q['inputs_after']
for name,h in q['inputs_before'].items():assert rows['quick/source/'+name]['sha256']==h
for name,v in q['products'].items():assert rows['quick/'+name]['sha256']==v['sha256'] and rows['quick/'+name]['bytes']==v['bytes']
assert ob['terminal'] and ob['actual_retained_translations'] and ob['accepted_call_boundary_observation'] and ob['declarations_excluded_from_call_counts'] and not ob['new_build'] and not ob['performance_claim']
for label in ['baseline','candidate']:
 raw=data('quick/products/'+label+'.c');text=raw.decode()
 for name,facts in ob['functions'][label].items():
  hits=list(re.finditer(r'^static \w+ (nl_'+name+r')\([^\n]*\) \{$',text,re.M));assert len(hits)==1;start=hits[0].start();body=text[start:text.index('\n}\n',start)+3];assert body.encode()==data('quick/'+label+'-'+name+'.c') and sha(body.encode())==facts['function_sha256'] and sha(raw)==facts['product_sha256']
  for target,count in facts['calls'].items():assert count==len(re.findall(r'\b\w*_'+target+r'\(',body.split('\n',1)[1]))
oldcalls=ob['functions']['baseline'];newcalls=ob['functions']['candidate'];assert oldcalls['check_function']['calls']['env_copy_symbols']==newcalls['check_function']['calls']['env_copy_symbols']==1
assert oldcalls['check_function']['calls']['check_block']==1 and newcalls['check_function']['calls']['check_block']==0 and newcalls['check_function']['calls']['check_owned_statement_list']==1
assert oldcalls['check_statement_list']['calls']['env_copy_symbols']==2 and newcalls['check_statement_list']['calls']['env_copy_symbols']==newcalls['check_owned_statement_list']['calls']['env_copy_symbols']==1
assert oldcalls['check_block']['function_sha256']==newcalls['check_block']['function_sha256']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git_data(commit,name):
 p.stdin.write((commit+':'+name+'\n').encode());p.stdin.flush();head=p.stdout.readline().split();assert len(head)==3 and head[1]==b'blob',name;raw=p.stdout.read(int(head[2]));assert p.stdout.read(1)==b'\n';return raw
assert len(q['source_before'])==3903
for name,h in q['source_before'].items():assert sha(git_data(m['source_commit'],name))==h,name
old=git_data(m['baseline_source_commit'],'src_nano/typecheck.nano');new=data('quick/source/src_nano/typecheck.nano');assert data('quick/source/src_nano/typecheck_baseline.nano')==old
oldbody=old[old.index(b'fn check_statement_list('):old.index(b'\nshadow check_block_value')];newbody=new[new.index(b'fn check_owned_statement_list('):new.index(b'\nshadow check_owned_statement_list')];assert newbody.replace(b'fn check_owned_statement_list(',b'fn check_statement_list(',1).replace(b'let mut local_symbols: array<Symbol> = symbols',b'let mut local_symbols: array<Symbol> = (env_copy_symbols symbols)',1).rstrip()==oldbody.rstrip()
asserts=lambda raw:[x for x in raw.splitlines() if x.lstrip().startswith(b'assert ')];it=iter(asserts(new));assert len(asserts(old))==pre['original_assertions']==839 and all(any(x==v for x in it) for v in asserts(old))
fixture=data('quick/source/tests/typecheck_scope_copy.nano');assert data('quick/source/tests/typecheck_scope_copy_baseline.nano')==fixture.replace(b'import "src_nano/typecheck.nano"',b'import "src_nano/typecheck_baseline.nano"',1)
recovered=fixture[:fixture.index(b'fn scope_body_tables(')]+fixture[fixture.index(b'fn scope_report_octets('):];recovered=recovered.replace(b'    (scope_body_tables)\n',b'').replace(b' (scope_body_tables) }',b' }');assert recovered==git_data(m['baseline_source_commit'],'tests/typecheck_scope_copy.nano')
assert git_data(m['baseline_source_commit'],'tests/typecheck_scope_copy_cases.json')==data('quick/source/tests/typecheck_scope_copy_cases.json')
f=js('first-fixture-refusal/quick-result.json');fs=js('first-fixture-refusal/result.json');assert f['terminal'] and not f.get('accepted_c_native_vm_boundary') and len(f['commands'])==1 and f['commands'][0]['exit']==1 and 'AssertionError' in f['exception'];assert b'tc_enclosing_function' in data('first-fixture-refusal/quick-001.stderr') and b'UNDEFINED VARIABLE' in data('first-fixture-refusal/quick-001.stderr')
for group in ['source','images','tools','hosts']:
 assert fs[group+'_before']==fs[group+'_after']
 for name,h in fs[group+'_before'].items():assert rows['first-fixture-refusal/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
for name,h in fs['source_before'].items():
 if name not in fs['overlays']:assert sha(git_data(m['baseline_source_commit'],name))==h
assert not js('quick/scope-observer-first-refusal.json')['accepted_observation'] and data('quick/inspect_scope_calls.py')!=data('quick/inspect_scope_calls_v2.py')!=data('quick/inspect_scope_calls_v3.py')
p.stdin.close();assert p.wait()==0
print('PASS %d records: exact source/839 preserved assertions/original fixture and24 records;108 baseline-candidate C-native+VM commands, selected guards and actual retained C scope-call boundary. First fixture refusal and observer revisions retained. Full-five-route SAN/compiler/factory/bootstrap/paired/pure-translator/release acceptance remains separate; no cause/gain claim.'%len(rows))
