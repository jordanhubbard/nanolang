"""I preserve a failed comparison without promoting a passing neighbor."""
from pathlib import Path
import sys,json,hashlib,re,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'d22a3e68664b7de7f3645f6b7f9e47e5eade80a3ec76e053601a794a544fcbbe');sha=lambda b:hashlib.sha256(b).hexdigest();pin='8fa1025477ca2b7db9266f1019b7192de20175d1';name='src_nano/typecheck.nano'
assert m['issue']==1263 and m['source_commit']==pin and m['diagnostic_only'] and m['failed_comparison']
assert not any(m[k] for k in ['component_acceptance','c_product_repair','new_bootstrap','native_acceptance','paired','whole_release','performance_claim'])
q=js('failed/qualification-result.json');binding=js('failed/binding.json');assert q['terminal'] and not q['accepted_component'] and not q['all_guards_match'] and len(q['commands'])==5
assert q['ordinary_seconds']==10 and q['build_seconds']==600 and q['run_seconds']==60
assert [r['label'] for r in q['commands']]==['baseline-fresh-admission','baseline-c-seed-build','baseline-c-producer-build','baseline-c-producer-verify','baseline-c-producer-translation']
assert [r['exit'] for r in q['commands']]==[0,0,0,0,1]
for i,row in enumerate(q['commands'],1):
 assert row['number']==i and not row.get('timed_out') and row['timeout'] in [60,600] and not row['cleanup']['remaining']
 assert row['stdout_sha256']==rows['failed/command-%03d.stdout'%i]['sha256'] and row['stderr_sha256']==rows['failed/command-%03d.stderr'%i]['sha256']
refusal=b'I cannot convert aggregate storage string to optional';assert refusal in data('failed/command-005.stderr') and b'791257/732225' in data('failed/command-005.stderr')
assert q['binding_sha256']==rows['failed/binding.json']['sha256'] and q['driver_sha256']==q['driver_after']==rows['failed/qualify.py']['sha256']
assert not q['owned_cleanup']['remaining'] and q['roles']['baseline']['all_guards_match'] and q['roles']['baseline']['fixture_guards_match']
assert not q['roles']['candidate']['fixture_guards_match'] and q['roles']['candidate']['fixture_after'] is None and not q['roles']['candidate']['fixture_reconstruction_after'] and not q['roles']['candidate']['products']
assert not q['roles']['baseline']['reports'] and not q['roles']['baseline']['controls']
pm,pr,pd,pj=read_archive(base.parent/'emission-function-index-cold-bootstrap-linux-20261126','53b44bb6e4b13a351b12bd633d0c57df8211b9771cf83d2e568a9c8b0bd268b3');origin=pj('bootstrap/bootstrap-result.json');receipt=pj('bootstrap/source/bin/nanoc_bootstrap.json');assert origin['source_pin']==pin and origin['original_make_pass'] and receipt['raw_stage1_stage2_equal']
for role in ['baseline','candidate']:
 prefix='failed/'+role+'/';s=js(prefix+'result.json');v=q['roles'][role];a=js(prefix+'admission-result.json');assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and a['accepted']
 assert len(s['source_before'])==3971 and s['overlays']=={name:binding[role+'_file_sha256']} and rows[prefix+'overlay.nano']['sha256']==rows[prefix+'source/'+name]['sha256']==binding[role+'_file_sha256']
 assert s['source_before']==dict(origin['source_after'],**s['overlays']) and s['producer_sha256']==receipt['artifacts']['nanoc_stage2']==rows[prefix+'source/bin/nanoc_emitter_phase']['sha256']
 for group in ['source','images','tools','hosts']:
  assert s[group+'_before']==s[group+'_after']==v[group+'_after']
  for p,h in v[group+'_after'].items():assert rows[prefix+'source/'+p if group in ['source','images'] else 'external/'+p.lstrip('/')]['sha256']==h
 for p,h in v['products'].items():assert rows[prefix+p]['sha256']==h['sha256']
 control=js(prefix+'cleanup-control-v2.json');assert control['accepted'] and not control['cleanup']['remaining']
assert rows['failed/baseline/products/c-producer.nvm']['sha256']=='a094d39de11b0351b5cfe8e416abb41550e8f9e8d8520ab5c5b795c52e436ae8' and rows['failed/baseline/products/c-producer.nvm']['bytes']==814560
original=data('failed/original.nano');braw=data('failed/baseline.nano');craw=data('failed/candidate.nano');shared=data('failed/shared.nano.txt');glob=data('failed/globals.nano.txt')
def function(raw,n):
 start=re.search(rb'(?m)^fn '+n.encode()+rb'\(',raw).start();end=re.search(rb'(?m)^(?:(?:pub )?fn |shadow |struct |enum |let (?:mut )?)',raw[start+3:]);return raw[start:start+3+end.start()] if end else raw[start:]
assert sha(original)==binding['original_sha256'] and sha(braw)==binding['baseline_file_sha256'] and sha(craw)==binding['candidate_file_sha256'] and sha(shared)==binding['shared_sha256'] and sha(glob)==binding['globals_sha256']
assert binding['candidate_to_baseline_reversal_exact'] and binding['baseline_to_original_reversal_exact'] and binding['all_original_assertions_preserved']
back=craw
for fn in binding['changed_production_functions']:back=back.replace(function(craw,fn),function(braw,fn),1)
assert back==braw
back=braw.replace(shared,b'',1).replace(glob,b'',1)
for fn in ['typecheck_advisory_core','check_legacy_par']:
 header=function(original,fn).splitlines()[0]+b'\n';assert back.count(header+b'    (tc_global_prefix_clear)\n')==1;back=back.replace(header+b'    (tc_global_prefix_clear)\n',header,1)
assert back==original
copied=function(original,'typecheck_advisory_core').replace(b'fn typecheck_advisory_core(',b'fn typecheck_global_prefix_body(',1);anchor=b'    /* Phase 2: NSType check each function body */';assert copied.count(anchor)==1;copied=copied.replace(anchor,b'    (tc_global_prefix_begin symbols)\n'+anchor,1);assert [i for i,l in enumerate(copied.splitlines(),1) if l and not l.strip()]==binding['copied_body_normalized_blank_lines']==[46,106];copied=re.sub(rb'(?m)^[ \t]+$',b'',copied);assert copied in shared
controls=js('failed/controls.json');fixture=data('failed/baseline/source/tests/typecheck_scope_copy.nano');harness=fixture.replace(b'import "src_nano/typecheck.nano"\n',b'',1);calls=b''
for i,body in enumerate(controls.values()):
 fn='_qualification_prefix_control_'+str(i);calls+=('    ('+fn+')\n').encode();harness+=('\nfn '+fn+'() -> void {\n'+body+'}\nshadow '+fn+' { ('+fn+') }\n').encode();assert ('\nshadow '+list(controls)[i]+' {\n'+body+'}\n').encode() in shared
harness=harness.replace(b'fn main() -> int {\n',b'fn main() -> int {\n'+calls,1);assert sha(fixture)==binding['original_fixture_sha256'] and harness==data('failed/harness.nano') and sha(harness)==binding['harness_sha256'];assert data('failed/baseline/fixture.nano')==braw+harness
corpus=js('failed/baseline/source/tests/typecheck_scope_copy_cases.json');assert len(corpus['cases'])==24 and corpus==js('failed/candidate/source/tests/typecheck_scope_copy_cases.json')
r=js('retained-controls/result.json');assert r['terminal'] and r['accepted_controls'] and r['all_parent_files_match'] and r['parent_before']==r['parent_after'] and not r['owned_cleanup']['remaining'] and len(r['commands'])==2
for command in r['commands']:assert command['exit']==0 and command['timeout']==60 and data('retained-controls/'+command['label']+'.stdout')==b'SCOPE-COPY-OK\n' and not command['cleanup']['remaining']
for label in ['trace-1','trace-2','trace-3']:
 t=js(label+'/result.json');assert t['terminal'] and t['exit']==0 and not t.get('timed_out') and t['all_guards_match'] and t['read_only_debugger'] and not t['inferior_function_calls'] and not t['c_product_edit'] and not t['owned_cleanup']['remaining']
 for key in ['translator','module','debugger','commands','prior']:assert t[key+'_before']==t[key+'_after']
 assert t['module_after']==rows['failed/baseline/products/c-producer.nvm']['sha256'] and t['translator_after']==rows['failed/baseline/source/bin/nvm2c']['sha256']
 assert rows[label+'/trace.gdb']['sha256']==t['commands_after'] and sha(data(label+'/prior-output.c'))==t['prior_after']
 assert refusal in data(label+'/stderr') and b'exited with code 01' in data(label+'/stdout')
assert b'exact = 1, alias_view = 0, optional_payload = 0' in data('trace-2/stdout') and b'final = 0' in data('trace-2/stdout') and b'kind = NVM_SHAPE_STRING' in data('trace-2/stdout') and b'kind = NVM_SHAPE_OPTIONAL' in data('trace-2/stdout')
p=js('trace-3/shape-summary.json');assert p['read_only'] and not p['inferior_calls'] and 'exception' not in p and p['ids']['source']==791257 and p['ids']['target']==732225 and p['track_shapes']==1 and not p['canonical_records'];assert any(v['function_name']=='tc_global_prefix_begin' and v['offset']==87 and 'target' in v['labels'] for v in p['matches']['outputs']);assert any(v['slot']==18 and 'conversion_source' in v['labels'] for v in p['matches']['globals'])
d=js('disassembly/result.json');assert d['exit']==0 and d['module_before']==d['module_after']==rows['failed/baseline/products/c-producer.nvm']['sha256'] and d['dump_sha256']==rows['disassembly/module.nasm']['sha256'];assert b'.function tc_global_prefix_begin 1 3 0 void 0\n' in data('disassembly/module.nasm')
for label in ['key-neighbors','vector-neighbor']:
 n=js(label+'/result.json');bd=js(label+'/binding.json');assert n['terminal'] and n['accepted_observation'] and n['all_parent_files_match'] and n['parent_before']==n['parent_after'] and n['subject_guards_match'] and not n['component_acceptance'] and not n['owned_cleanup']['remaining']
 assert n['binding_sha256']==rows[label+'/binding.json']['sha256'] and bd['original_fixture_sha256']==rows['failed/baseline/fixture.nano']['sha256']
 s=js(label+'/subject/result.json');assert s['admitted_setup'] and s['all_guards_match'] and not s['overlays'] and s['source_before']==origin['source_after']
 for group,values in n['guards_after'].items():
  assert values==s[group+'_after']
  for p,h in values.items():assert rows[label+'/subject/source/'+p if group in ['source','images'] else 'external/'+p.lstrip('/')]['sha256']==h
 for variant,v in n['variants'].items():
  z=bd['variants'][variant];assert z['reversal_exact'] and z['all_assertions_preserved'];raw=data(label+'/'+variant+'/fixture.nano');assert sha(raw)==z['source_sha256']==v['fixture_before']==v['fixture_after'];back=raw
  for change in reversed(z.get('patches',[dict(old=z.get('old'),new=z.get('new'))])):back=back.replace(change['new'].encode(),change['old'].encode(),1)
  assert back==data('failed/baseline/fixture.nano')
  assert v['module_sha256']==rows[label+'/'+variant+'/products/module.nvm']['sha256'] and v['output_sha256']==rows[label+'/'+variant+'/products/output.c']['sha256']
  assert data(label+'/'+variant+'-vm.stdout')==b'SCOPE-COPY-OK\n' and v['translation_ready']==(v['translation_exit']==0)
  if label=='key-neighbors':assert v['translation_exit']==1 and v['output_sha256']==v['prior_sha256'] and refusal in data(label+'/'+variant+'-translation.stderr')
  else:assert v['translation_exit']==0 and v['output_sha256']!=v['prior_sha256']
 for command in n['commands']:
  assert command['timeout'] in [60,600] and not command.get('timed_out') and not command['cleanup']['remaining'];assert command['stdout_sha256']==rows[label+'/'+command['label']+'.stdout']['sha256'] and command['stderr_sha256']==rows[label+'/'+command['label']+'.stderr']['sha256']
reject=js('collector-source/parent-refusal.json');assert reject['terminal'] and reject['accepted'] and reject['exit']==1 and reject['product_commands']==0 and not reject['source_created'];assert reject['parent_sha256']==rows['failed/qualification-result.json']['sha256']==rows['collector-source/failed-parent.json']['sha256'] and reject['helper_sha256']==rows['collector-source/reject_failed_parent.py']['sha256'] and b'AssertionError' in data('collector-source/parent-refusal.stderr')
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();assert original==subprocess.check_output(['git','show',pin+':'+name],cwd=repo)
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for role in ['baseline','candidate']:
 s=js('failed/'+role+'/result.json')
 for path,h in s['source_before'].items():
  if path in s['overlays']:continue
  proc.stdin.write((pin+':'+path+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,path
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: failed five-command comparison and exact source/fixture reversals; retained C-seed/VM runtime controls, three read-only C traces, typed-key refusals, vector VM/translation readiness, prior outputs, full guards and failed-parent bootstrap refusal. No component, C repair, sanitizer/native, paired or release acceptance.'%len(rows))
