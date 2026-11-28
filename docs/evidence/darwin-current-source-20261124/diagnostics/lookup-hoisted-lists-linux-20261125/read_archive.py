"""I check the exact hoisted-list comparison and retain the private-fixture refusal."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'39d7e5b3170b24d2a76e3d1746b4b88f3d24a9c414e3361a7d73913b7e51ea8c')
sha=lambda b:hashlib.sha256(b).hexdigest()
s=js('comparison/result.json');q=js('comparison/quick-result.json');audit=js('comparison/c-audit.json');bad=js('private-fixture-refusal/quick-result.json');post=js('private-fixture-post-terminal-audit.json')
assert m['source_commit']==s['source_base']=='7b9f08ee0941c23810d18fa3f8ebfbb759c968de'
assert m['c_native_vm_only'] and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release','historical_cause','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['overlays'] and not s['working_overlay']
assert q['terminal'] and q['accepted_c_native_vm_boundary'] and q['all_guards_match'] and q['ordinary_seconds']==10 and len(q['commands'])==108 and len(q['cases'])==24
assert q['source_base']==m['baseline_source_commit']=='95a4062efebe0e97d8cde271117eac1deab21595'
for c in q['commands']:assert c['exit']==0 and not c.get('timed_out') and c['timeout'] in [60,600]
for n in [2,5,8,11]:assert data('comparison/quick-%03d.stdout'%n)==b'LOOKUP-PROJECTION-OK\n'
corpus=js('comparison/source/tests/typecheck_scope_copy_cases.json')['cases'];wanted=js('comparison/source/tests/nanoisa_lookup_projection_cases.json')
assert len(corpus)==len(wanted)==24
for i,(source,expected,rec) in enumerate(zip(corpus,wanted,q['cases'])):
 raw=bytes.fromhex(source['source_hex']);assert sha(raw)==expected['source_sha256']==rec['source_sha256']
 assert data('comparison/diagnostics/case-%02d.nano'%i)==raw
 reports=[data('comparison/quick-%03d.stdout'%n) for n in rec['commands']]
 assert len(reports)==4 and all(r==reports[0] for r in reports) and sha(reports[0])==expected['report_sha256']==rec['expected_sha256']
for group in ['source','images','tools','hosts','inputs']:
 assert q[group+'_before']==q[group+'_after']
 for name,h in q[group+'_before'].items():
  key='comparison/source/'+name if group in ['source','images','inputs'] else 'external/'+name.lstrip('/')
  assert rows[key]['sha256']==h
assert audit['terminal'] and audit['accepted_static_comparison'] and not audit['new_execution'] and not audit['performance_claim']
assert audit['original_assertion_lines']==1641 and audit['new_assertion_lines']==5
for name,pair in {'nisa_fn_index':(2,1),'nisa_call_index':(9,7),'nisa_enum_index':(2,1)}.items():
 for label,expected in zip(['baseline','candidate'],pair):
  facts=audit['functions'][label][name];raw=data('comparison/products/'+label+'.c');text=raw.decode()
  matches=[v for v in re.finditer(r'^static[^\n]*\b(nl_\w*'+name+r')\([^\n]*\) \{$',text,re.M) if 'shadow_' not in v[1]]
  assert len(matches)==1;start=matches[0].start();body=text[start:text.index('\n}\n',start)+3]
  assert body.encode()==data('comparison/'+label+'-'+name+'.c') and sha(raw)==facts['product_sha256'] and sha(body.encode())==facts['function_sha256']
  assert facts['parser_projection_copies']==len(re.findall(r'\br\[\d+\]\s*=\s*rl\[0\];',body))==expected
assert b'fputs' in data('comparison/candidate-nisa_fn_index.c') and b'fflush(stdout)' in data('comparison/candidate-nisa_fn_index.c')
assert bad['terminal'] and len(bad['commands'])==7 and bad['commands'][-1]['exit']==1 and not bad['commands'][-1].get('timed_out')
assert all(c['exit']==0 for c in bad['commands'][:-1])
err=data('private-fixture-refusal/quick-007.stderr');assert err.count(b'-- E009 PRIVATE ACCESS')==5 and b'parser_init_ast_lists' in err and b'parser_store_enum' in err
assert post['terminal'] and post['post_terminal'] and post['read_only'] and post['all_selected_guards_match'] and post['receipt_sha256']==rows['private-fixture-refusal/quick-result.json']['sha256']
for group in ['source','images','tools','hosts','inputs']:
 assert post[group+'_after']==bad[group+'_before']
 for name,h in bad[group+'_before'].items():
  key='private-fixture-refusal/source/'+name if group in ['source','images','inputs'] else 'external/'+name.lstrip('/')
  assert rows[key]['sha256']==h
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git(commit,name):
 proc.stdin.write((commit+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
 raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for label,pin in [('comparison',m['source_commit']),('private-fixture-refusal',m['failed_fixture_source_commit'])]:
 setup=js(label+'/result.json')
 for name,h in setup['source_before'].items():assert sha(git(pin,name))==h,(label,name)
new=data('comparison/source/src_nano/compiler/nanoisa_codegen.nano').decode();old=git(m['baseline_source_commit'],'src_nano/compiler/nanoisa_codegen.nano').decode()
assertions=lambda text:[line for line in text.splitlines() if re.search(r'\bassert\b',line)]
previous=assertions(old);current=assertions(new);it=iter(current);assert len(previous)==1641 and len(current)==1646 and all(any(line==v for v in it) for line in previous)
assert 'parser_store_enum' not in new and 'parser_init_ast_lists' not in new
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: baseline95a/candidate7b9, all108 C-native/VM commands and24 complete reports, generated-C projection sites2->1/9->7/2->1, every prior assertion plus five enum boundaries and retained five E009 private-fixture refusals. No full-five/current-native/paired/release or cause/gain claim.'%len(rows))
