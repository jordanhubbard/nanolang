"""I bind both lookup implementations to their original complete lowering reports."""
from pathlib import Path
import hashlib,json,subprocess,sys,re
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
sha=lambda b:hashlib.sha256(b).hexdigest()
m,rows,data,js=read_archive(base,'ad2870976cf26941a50ee7b392e8238b81c9612b5b4a71376658f80a5eed192a')
q=js('qualified/qualification-result.json');binding=js('qualified/binding.json');name='src_nano/compiler/nanoisa_codegen.nano';pin='771a14330475bcd784d7cb19a215d6a4f15728d4';producer='8d7c240099edbe11cde499a268705f34660c2d940da4895378fc2a2961c886a5'
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
pm,pr,pd,pj=read_archive(base.parent/'emitter-record-list-cold-bootstrap-linux-20261126','d230b2bd730561baa99e3496e8c58eb14fd570e24176c90bba9cb255f9df9f8b')
origin=pj('bootstrap/bootstrap-result.json');installed=pj('bootstrap/source/bin/nanoc_bootstrap.json')
assert origin['terminal'] and origin['original_make_pass'] and origin['exit']==0 and origin['source_pin']==pin and installed['phase']=='stage2' and installed['raw_stage1_stage2_equal'] and len(installed['steps'])==17 and all(v['exit_code']==0 for v in installed['steps'])
assert installed['artifacts']['nanoc_stage2']==pr['bootstrap/source/bin/nanoc_stage2']['sha256']==producer
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
original=subprocess.check_output(['git','show',pin+':'+name],cwd=repo)
assert binding['path']==name and binding['original_sha256']==sha(original) and binding['candidate_to_baseline_reversal_exact'] and binding['baseline_to_original_reversal_exact'] and binding['all_original_assertions_preserved'] and binding['embedded_fixture']
assert binding['changed_production_functions']==['nisa_fn_index','nisa_call_index','nisa_emit_module_context','nisa_emit_shadows_native_inner']
def function(raw,name):
 start=raw.index(('fn '+name+'(').encode());end=raw.index(('\nshadow '+name+' {').encode(),start);return raw[start:end]
braw=data('qualified/baseline/source/'+name);craw=data('qualified/candidate/source/'+name)
assert data('qualified/original.nano')==original and data('qualified/baseline.nano')==braw and data('qualified/candidate.nano')==craw
reversed_candidate=craw
for function_name in binding['changed_production_functions']:reversed_candidate=reversed_candidate.replace(function(craw,function_name),function(original,function_name),1)
assert reversed_candidate==braw
shared=data('qualified/shared.nano.txt');globals_text=data('qualified/globals.nano.txt');reset=b'fn nisa_reset() -> int {\n'
assert sha(shared)==binding['shared_sha256'] and sha(globals_text)==binding['globals_sha256']
assert braw.replace(shared,b'',1).replace(globals_text,b'',1).replace(reset+b'    (nisa_function_index_clear)\n',reset,1)==original
for old_name,new_name,args in [('nisa_emit_module_context','nisa_emit_module_indexed_body','parser program bound context'),('nisa_emit_shadows_native_inner','nisa_emit_shadows_indexed_body','parser first_shadow bound context native_owner native_artifact')]:
 old_body=function(original,old_name);expected=old_body.replace(('fn '+old_name+'(').encode(),('fn '+new_name+'(').encode(),1)
 anchor=b'        let n: int = (parser_get_function_count parser)' if old_name=='nisa_emit_module_context' else b'    let initializers: string = (nisa_emit_globals parser)'
 insertion=b'        (nisa_function_index_begin parser.functions)\n' if old_name=='nisa_emit_module_context' else b'    (nisa_function_index_begin parser.functions)\n'
 assert expected.count(anchor)==1;expected=expected.replace(anchor,insertion+anchor,1)
 assert expected in shared and shared.count(('\nfn '+new_name+'(').encode())==1
 wrapper=old_body.splitlines()[0]+b'\n    (nisa_function_index_clear)\n    let output: string = ('+new_name.encode()+b' '+args.encode()+b')\n    (nisa_function_index_clear)\n    return output\n}\n'
 assert function(craw,old_name)==wrapper

harness=data('qualified/harness.nano');controls=js('qualified/controls.json')
original_fixture=data('qualified/baseline/source/tests/nanoisa_lookup_projection.nano')
assert sha(original_fixture)==binding['original_fixture_sha256'] and sha(harness)==binding['harness_sha256']
remove=b'from "src_nano/compiler/nanoisa_codegen.nano" import nanoisa_emit_nasm, nanoisa_last_lowering_error\n'
expected_harness=original_fixture.replace(remove,b'',1);calls=b''
for i,body in enumerate(controls.values()):
 fn='_qualification_index_control_'+str(i);calls+=('    ('+fn+')\n').encode();expected_harness+=('\nfn '+fn+'() -> void {\n'+body+'}\nshadow '+fn+' { ('+fn+') }\n').encode()
 assert ('\nshadow '+list(controls)[i]+' {\n'+body+'}\n').encode() in shared
needle=b'    let report: string = (lookup_lowering_report '
assert expected_harness.count(needle)==1 and expected_harness.replace(needle,calls+needle,1)==harness
for role,raw in [('baseline',braw),('candidate',craw)]:
 v=q['roles'][role];fixture=data('qualified/'+role+'/fixture.nano')
 assert fixture==raw+harness and sha(fixture)==v['fixture_before']==v['fixture_after'] and sha(harness)==v['harness_before']==v['harness_after'] and v['fixture_guards_match'] and v['fixture_reconstruction_exact'] and v['fixture_reconstruction_after']
for label,result in [('qualified',q)]:
 for role in ['baseline','candidate']:
  prefix=label+'/'+role+'/';s=js(prefix+'result.json');a=js(prefix+'admission-result.json');v=result['roles'][role]
  assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and s['working_overlay'] and s['source_base']==pin and not s['new_bootstrap'] and not s['stage_alias']
  assert s['source_before']==dict(origin['source_after'],**s['overlays'])
  assert len(s['source_before'])==3971 and len(s['overlays'])==1 and name in s['overlays'] and s['overlays'][name]==rows[prefix+'overlay.nano']['sha256']==rows[prefix+'source/'+name]['sha256']
  assert s['bootstrap_source_inputs']==origin['source_after'] and s['bootstrap_source_pin']==pin and s['bootstrap_result_sha256']==pr['bootstrap/bootstrap-result.json']['sha256'] and s['bootstrap_receipt_sha256']==pr['bootstrap/source/bin/nanoc_bootstrap.json']['sha256']
  assert s['bootstrap_publication']=='08c223f945474f459364c24d5e8767e4ede1b0ab' and s['producer_sha256']==v['producer_after']==rows[prefix+'source/bin/nanoc_emitter_phase']['sha256']==producer and s['hosts_before']==installed['hosts'] and not s['native_consumer_accepted']
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
refusal=js('collector-source/live-refusal-result.json');assert refusal['terminal'] and refusal['accepted'] and refusal['exit']==1 and not refusal['archive_created'] and not js('collector-source/live-qualification-result.json')['terminal'] and refusal['live_result_sha256']==rows['collector-source/live-qualification-result.json']['sha256']
assert b'AssertionError' in data('collector-source/live-refusal.stderr')
assert not any(n.endswith('/source/bin/nanoc_stage2') or n.endswith('/source/bin/nanoc_bootstrap.json') for n in rows)
assert original==subprocess.check_output(['git','show',binding['head']+':'+name],cwd=repo)
subprocess.run(['git','merge-base','--is-ancestor','08c223f945474f459364c24d5e8767e4ede1b0ab',binding['head']],cwd=repo,check=True)
assert js('collector-source/live-refusal-result.json')['collector_sha256']==rows['collector-source/package.py']['sha256']
materialization=js('collector-source/materialization.json');assert materialization['terminal'] and materialization['accepted_call_path_observation'] and materialization['actual_retained_translations'] and not materialization['new_build'] and not materialization['performance_claim']
for label,v in materialization['functions'].items():
 role,provider,function_name=v['role'],v['producer'],v['source_function'];raw=data('qualified/'+role+'/products/'+provider+'.c');body=data('collector-source/'+label+'.c.txt')
 assert sha(raw)==v['product_sha256'] and body in raw and sha(body)==v['function_sha256']
 assert v['calls']=={name:len(re.findall(r'\b\w*'+name+r'\(',body.decode())) for name in materialization['call_names']}
 if function_name in ['nisa_fn_index','nisa_call_index']:assert v['calls']['nisa_function_index_first']==(1 if role=='candidate' else 0)
 if function_name in ['nisa_emit_module_context','nisa_emit_shadows_native_inner']:assert v['calls']['nisa_function_index_clear']==(2 if role=='candidate' else 0)
p=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for role in ['baseline','candidate']:
 s=js('qualified/'+role+'/result.json')
 for path,h in s['source_before'].items():
  if path in s['overlays']:continue
  p.stdin.write((pin+':'+path+'\n').encode());p.stdin.flush();head=p.stdout.readline().split();assert len(head)==3 and head[1]==b'blob';raw=p.stdout.read(int(head[2]));assert p.stdout.read(1)==b'\n' and sha(raw)==h,path
p.stdin.close();assert p.wait()==0

bootstrap_refusal=js('collector-source/bootstrap-nonterminal-refusal.json')
assert bootstrap_refusal['terminal'] and bootstrap_refusal['accepted'] and bootstrap_refusal['exit']==1 and bootstrap_refusal['product_commands']==0 and not bootstrap_refusal['source_created']
assert not js('collector-source/bootstrap-nonterminal-parent.json')['terminal']
assert bootstrap_refusal['parent_sha256']==rows['collector-source/bootstrap-nonterminal-parent.json']['sha256'] and bootstrap_refusal['generator_sha256']==rows['collector-source/prepare_bootstrap.py']['sha256']
assert b'AssertionError' in data('collector-source/bootstrap-nonterminal-refusal.stderr')
assert js('qualified/shadow-policy.json')['exit']==0
print('PASS %d records: exact indexed and retained implementations; common assertions, five routes each,270 commands/240 original reports/10 controls; embedded fixture reconstruction, source reversals, published bootstrap origin, guards, cleanup and actual nonterminal refusal. Component only.'%len(rows))
