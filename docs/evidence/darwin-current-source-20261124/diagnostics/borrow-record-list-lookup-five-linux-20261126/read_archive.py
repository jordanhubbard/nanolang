"""I bind both lookup implementations to their original complete lowering reports."""
from pathlib import Path
import hashlib,json,subprocess,sys,re
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'d026144adeecac9a845871aca20d1144f1f7ff182375296bed8b88744c4ae61c')
q=js('qualified/qualification-result.json');bad=js('refused/qualification-result.json');binding=js('qualified/binding.json');name='src_nano/compiler/nanoisa_borrows.nano';pin='7b9f08ee0941c23810d18fa3f8ebfbb759c968de';producer='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
assert m['source_commit']==binding['source_base']==pin and m['component_only'] and m['full_five_routes']
assert not any(m[k] for k in ['permanent_unittest_invoked','updated_compiler_acceptance','new_bootstrap','factory_acceptance','paired_acceptance','whole_release','performance_claim'])
assert q['terminal'] and q['accepted_component'] and q['all_guards_match'] and q['component_only'] and q['full_five_routes'] and not q['new_bootstrap'] and not q['current_native_acceptance'] and not q['whole_release']
assert q['ordinary_seconds']==10 and q['build_seconds']==600 and q['run_seconds']==60 and len(q['commands'])==270
assert q['binding_sha256']==rows['qualified/binding.json']['sha256'] and q['driver_sha256']==q['driver_after']==rows['qualified/qualify.py']['sha256'] and q['cleanup_helper_sha256']==q['cleanup_helper_after']==rows['qualified/baseline/owned_cleanup.py']['sha256']
assert q['owned_cleanup']['terminal'] and q['owned_cleanup']['owned_only'] and q['owned_cleanup']['pidfd_identity'] and not q['owned_cleanup']['remaining']
for i,c in enumerate(q['commands'],1):
 assert c['number']==i and c['exit']==0 and not c.get('timed_out') and c['timeout'] in [60,600] and not c['cleanup']['remaining'] and not c['cleanup']['records']
 assert c['stdout_sha256']==rows['qualified/command-%03d.stdout'%i]['sha256'] and c['stderr_sha256']==rows['qualified/command-%03d.stderr'%i]['sha256']
route_names=['c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native']
pm,pr,pd,pj=read_archive(base.parent/'lookup-hoisted-cold-bootstrap-linux-20261125','d232ed852a6f375dc76f1d34cac02b22c87aa05c5b220073180f8dbbe39d7e49')
origin=pj('bootstrap/bootstrap-result.json');installed=pj('bootstrap/source/bin/nanoc_bootstrap.json')
assert origin['terminal'] and origin['original_make_pass'] and origin['exit']==0 and origin['source_pin']==pin and installed['phase']=='stage2' and installed['raw_stage1_stage2_equal'] and len(installed['steps'])==17 and all(v['exit_code']==0 for v in installed['steps'])
assert installed['artifacts']['nanoc_stage2']==pr['bootstrap/source/bin/nanoc_stage2']['sha256']==producer
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
original=subprocess.check_output(['git','show',pin+':'+name],cwd=repo)
assert binding['path']==name and binding['original_file_sha256']==sha(original) and binding['only_production_function_changed']=='nb_record' and binding['baseline_reversal_exact'] and binding['all_previous_bytes_and_assertions_preserved']
oldfn=data('qualified/baseline-function.nano.txt');newfn=data('qualified/candidate-function.nano.txt');shadow=data('qualified/added-shadow.nano.txt')
assert sha(oldfn)==binding['baseline_function_sha256'] and sha(newfn)==binding['candidate_function_sha256'] and sha(shadow)==binding['added_shadow_sha256']
assert oldfn in original and b'parser_get_struct_def_count parser' in oldfn and b'parser_get_struct_def parser i' in oldfn and b'list_ASTStruct_length records' in newfn and b'list_ASTStruct_get records i' in newfn
parser_source=data('qualified/baseline/source/src_nano/parser.nano')
for file,expected in [('parser-count-getter.nano.txt',b'pub fn parser_get_struct_def_count(p: Parser) -> int {\n    return (list_ASTStruct_length p.structs)\n}'),('parser-record-getter.nano.txt',b'pub fn parser_get_struct_def(p: Parser, idx: int) -> ASTStruct {\n    return (list_ASTStruct_get p.structs idx)\n}')]:
 assert data('collector-source/'+file)==expected and expected in parser_source
braw=data('qualified/baseline/source/'+name);craw=data('qualified/candidate/source/'+name)
assert braw.replace(shadow,b'',1)==original and craw.replace(newfn,oldfn,1)==braw
assert len(shadow)>0 and oldfn.count(b'fn nb_record(')==newfn.count(b'fn nb_record(')==1
for label,result in [('qualified',q),('refused',bad)]:
 for role in ['baseline','candidate']:
  prefix=label+'/'+role+'/';s=js(prefix+'result.json');a=js(prefix+'admission-result.json');v=result['roles'][role]
  assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and s['working_overlay'] and s['source_base']==pin and not s['new_bootstrap'] and not s['stage_alias']
  assert s['source_before']==dict(origin['source_after'],**s['overlays'])
  assert len(s['source_before'])==3911 and len(s['overlays'])==1 and name in s['overlays'] and s['overlays'][name]==rows[prefix+'overlay.nano']['sha256']==rows[prefix+'source/'+name]['sha256']
  assert s['bootstrap_source_inputs']==origin['source_after'] and s['bootstrap_source_pin']==pin and s['bootstrap_result_sha256']==pr['bootstrap/bootstrap-result.json']['sha256'] and s['bootstrap_receipt_sha256']==pr['bootstrap/source/bin/nanoc_bootstrap.json']['sha256']
  assert s['bootstrap_publication']=='cac73fa60c22d33e45afa3dd8b8ced7220f8b804' and s['producer_sha256']==v['producer_after']==rows[prefix+'source/bin/nanoc_emitter_phase']['sha256']==producer and s['hosts_before']==installed['hosts'] and not s['native_consumer_accepted']
  assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3 and all(c['cpu']>=1 and c['share']>=.75 for c in s['cpu_rounds'])
  for group in ['source','images','tools','hosts']:
   assert s[group+'_before']==s[group+'_after']==v[group+'_after']
   for path,h in s[group+'_before'].items():assert rows[prefix+'source/'+path if group in ['source','images'] else 'external/'+path.lstrip('/')]['sha256']==h
  for path,h in s['images_before'].items():assert pr['bootstrap/source/'+path]['sha256']==h
  for path,h in v['products'].items():assert rows[prefix+path]['sha256']==h['sha256'] and rows[prefix+path]['bytes']==h['bytes']
  control=js(prefix+'cleanup-control-v2.json');assert control['terminal'] and control['accepted'] and control['helper_sha256']==rows[prefix+'owned_cleanup.py']['sha256'] and control['control_sha256']==rows[prefix+'cleanup_control_v2.py']['sha256'] and not control['cleanup']['remaining'] and any(c['waitpid']==[control['child']['pid'],9] for c in control['cleanup']['records'])
for role in ['baseline','candidate']:
 prefix='qualified/'+role+'/';s=js(prefix+'result.json');v=q['roles'][role]
 assert v['accepted_component'] and v['all_guards_match'] and v['setup_sha256']==rows[prefix+'result.json']['sha256'] and v['admission_sha256']==rows[prefix+'admission-result.json']['sha256'] and s['overlays'][name]==binding[role+'_file_sha256']
 corpus=js(prefix+'source/tests/typecheck_scope_copy_cases.json')['cases'];expected=js(prefix+'source/tests/nanoisa_lookup_projection_cases.json');assert len(corpus)==len(expected)==24
 assert len(v['controls'])==5 and [c['route'] for c in v['controls']]==route_names and len(v['reports'])==120
 for c in v['controls']:assert data('qualified/command-%03d.stdout'%c['command'])==b'LOOKUP-PROJECTION-OK\n'
 assert [(r['index'],r['route']) for r in v['reports']]==[(i,n) for i in range(24) for n in route_names]
 for report in v['reports']:
  i=report['index'];raw=bytes.fromhex(corpus[i]['source_hex']);assert sha(raw)==expected[i]['source_sha256']==report['source_sha256'] and data(prefix+'diagnostics/case-%02d.nano'%i)==raw
  assert report['report_sha256']==expected[i]['report_sha256']==rows['qualified/command-%03d.stdout'%report['command']]['sha256']
 commands=[c for c in q['commands'] if c['label'].startswith(role+'-')];assert len(commands)==135
 assert [c['label'] for c in commands[:10]]==[role+'-'+n for n in ['fresh-admission','c-seed-build','c-producer-build','c-producer-verify','c-producer-translation','c-producer-sanitized-build','nanolang-producer-build','nanolang-producer-verify','nanolang-producer-translation','nanolang-producer-sanitized-build']]
 for c in commands:
  assert c['cwd']==str(Path(m['roots']['qualified'])/role/'source')
  if c['label'].endswith('-build') or c['label'].endswith('-translation'):assert c['timeout']==600
  else:assert c['timeout']==60
  if c['label'].endswith('-sanitized-build'):
   for arg in ['-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all']:assert arg in c['argv']
  if c['label']==role+'-nanolang-producer-build':assert c['argv'][0]==s['producer'] and c['argv'][2]=='--emit-nvm'
assert bad['terminal'] and not bad['accepted_component'] and bad['all_guards_match'] and len(bad['commands'])==2 and bad['commands'][-1]['exit']==1
err=data('refused/command-002.stderr');assert b'E009 PRIVATE ACCESS' in err and b"private function 'parser_init_ast_lists'" in err and not any(v['products'] for v in bad['roles'].values())
badshadow=data('refused/added-shadow.nano.txt')
assert badshadow.replace(b'    let empty: Parser = (parser_init_ast_lists)',b'    let empty_tokens: List<LexerToken> = (tokenize_string "" "empty-lookup.nano" (list_CompilerDiagnostic_new))\n    let empty: Parser = (parse_program empty_tokens (list_LexerToken_length empty_tokens) "empty-lookup.nano")',1)==shadow
failure=js('preparation-refused/preparation-failure.json');assert failure['terminal'] and failure['product_commands']==0 and not failure['source_created']
assert not any('/source/' in r for r in rows if r.startswith('preparation-refused/'))
refusal=js('collector-source/live-refusal-result.json');assert refusal['terminal'] and refusal['accepted'] and refusal['exit']==1 and not refusal['archive_created'] and not js('collector-source/live-qualification-result.json')['terminal'] and refusal['live_result_sha256']==rows['collector-source/live-qualification-result.json']['sha256']
assert b'AssertionError' in data('collector-source/live-refusal.stderr')
assert not any(n.endswith('/source/bin/nanoc_stage2') or n.endswith('/source/bin/nanoc_bootstrap.json') for n in rows)
assert original==subprocess.check_output(['git','show',binding['head']+':'+name],cwd=repo)
subprocess.run(['git','merge-base','--is-ancestor','cac73fa60c22d33e45afa3dd8b8ced7220f8b804',binding['head']],cwd=repo,check=True)
resource=js('resource-consumer/consumer-result.json');assert resource['terminal'] and resource['accepted_resource_reports'] and resource['all_guards_match'] and resource['component_consumer_only'] and not resource['new_build'] and not resource['whole_release'] and resource['ordinary_seconds']==10 and resource['run_seconds']==60
assert resource['parent_result_sha256']==rows['qualified/qualification-result.json']['sha256'] and resource['driver_sha256']==resource['driver_after']==rows['resource-consumer/consume.py']['sha256'] and resource['helper_sha256']==resource['helper_after']==rows['resource-consumer/owned_cleanup.py']['sha256']
assert resource['parent_before']==resource['parent_after']
parent_files={n[len('qualified/'):]:{k:r[k] for k in ['sha256','bytes','mode']} for n,r in rows.items() if n.startswith('qualified/')}
parent_links={r['path'][len('qualified/'):]:r['target'] for r in m['symlinks'] if r['path'].startswith('qualified/')}
assert resource['parent_before']==dict(files=parent_files,links=parent_links)
expected_external={**js('qualified/baseline/result.json')['tools_after'],**js('qualified/baseline/result.json')['hosts_after']}
assert resource['external_before']==resource['external_after']==expected_external
assert resource['owned_cleanup']['terminal'] and resource['owned_cleanup']['owned_only'] and resource['owned_cleanup']['pidfd_identity'] and not resource['owned_cleanup']['remaining']
admission=js('resource-consumer/admission-result.json');assert admission['terminal'] and admission['accepted'] and admission['filesystem']!='nfs' and admission['free_bytes']>15*1024**3 and not admission['other_jobs'] and all(c['cpu']>=1 and c['share']>=.75 for c in admission['cpu_rounds'])
control=js('resource-consumer/cleanup-control-v2.json');assert control['accepted'] and not control['cleanup']['remaining'] and control['helper_sha256']==resource['helper_sha256'] and control['control_sha256']==rows['resource-consumer/cleanup_control_v2.py']['sha256'] and any(c['waitpid']==[control['child']['pid'],9] for c in control['cleanup']['records'])
assert resource['vm_sha256']==rows['resource-consumer/products/nano_vm']['sha256']==js('qualified/baseline/result.json')['images_after']['bin/nano_vm']
for path,h in resource['copied_products'].items():
 role=Path(path).parts[1];filename=Path(path).name;assert rows['resource-consumer/'+path]['sha256']==h==q['roles'][role]['products']['products/'+filename]['sha256']
fixture_names=['source_borrow_shared.nano','source_borrow_exclusive.nano','source_borrow_multi.nano','source_borrow_nested.nano','source_borrow_resource_paths.nano']
assert list(resource['fixtures'])==fixture_names and len(resource['commands'])==len(resource['reports'])==50
assert [(r['fixture'],r['role'],r['route']) for r in resource['reports']]==[(n,role,route) for n in fixture_names for role in ['baseline','candidate'] for route in route_names]
for name in fixture_names:
 raw=data('resource-consumer/inputs/'+name);assert sha(raw)==resource['fixtures'][name]==rows['qualified/baseline/source/tests/nanoisa/fixtures/'+name]['sha256']
 expected=None
 for report in [r for r in resource['reports'] if r['fixture']==name]:
  n=report['command'];command=resource['commands'][n-1];assert command['number']==n and command['exit']==0 and command['timeout']==60 and not command.get('timed_out') and not command['cleanup']['records'] and not command['cleanup']['remaining']
  raw=data('resource-consumer/command-%02d.stdout'%n);assert sha(raw)==report['sha256'] and len(raw)==report['bytes'] and b'.ownership "' in raw and raw.endswith(b'\nLOOKUP-ERROR:\n\n')
  if expected is None:assert report['role']=='baseline' and report['route']=='c-seed-native';expected=raw
  assert raw==expected
refusal=js('resource-consumer/nonterminal-parent-refusal.json');assert refusal['terminal'] and refusal['accepted'] and refusal['exit']==1 and refusal['product_commands']==refusal['products_copied']==refusal['inputs_copied']==0
assert not js('resource-consumer/nonterminal-parent.json')['terminal'] and refusal['parent_sha256']==rows['resource-consumer/nonterminal-parent.json']['sha256'] and refusal['driver_sha256']==resource['driver_sha256'] and b'AssertionError' in data('resource-consumer/nonterminal-parent-refusal.stderr')
assert js('collector-source/live-refusal-result.json')['collector_sha256']==rows['collector-source/package.py']['sha256']
materialization=js('collector-source/materialization.json');assert materialization['terminal'] and materialization['accepted_call_path_observation'] and materialization['actual_retained_translations'] and not materialization['new_build'] and not materialization['performance_claim']
for role in ['baseline','candidate']:
 for provider in ['c-producer','nanolang-producer']:
  label=role+'-'+provider;v=materialization['functions'][label];raw=data('qualified/'+role+'/products/'+provider+'.c');body=data('collector-source/'+label+'-nb_record.c.txt').decode()
  assert sha(raw)==v['product_sha256'] and body.encode() in raw and sha(body.encode())==v['function_sha256']
  assert v['calls']=={name:len(re.findall(r'\b\w*'+name+r'\(',body)) for name in ['parser_get_struct_def_count','parser_get_struct_def','list_ASTStruct_length','list_ASTStruct_get']}
  expected=1 if role=='baseline' else 0
  assert v['calls']['parser_get_struct_def_count']==v['calls']['parser_get_struct_def']==expected
p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for role in ['baseline','candidate']:
 s=js('qualified/'+role+'/result.json')
 for path,h in s['source_before'].items():
  if path in s['overlays']:continue
  p.stdin.write((pin+':'+path+'\n').encode());p.stdin.flush();head=p.stdout.readline().split();assert len(head)==3 and head[1]==b'blob';raw=p.stdout.read(int(head[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,path
p.stdin.close();assert p.wait()==0
print('PASS %d records: exact retained/manual and candidate lookup, unchanged prior bytes/assertions, same new useful shadows, five routes each, 270 commands/240 original complete reports/10 controls plus 50 resource reports without new builds, qualified provider origin, full guards and preserved private-access/preparer/live-collector refusals. Component only; no current compiler/factory/bootstrap/paired/speedup/release acceptance.'%len(rows))
