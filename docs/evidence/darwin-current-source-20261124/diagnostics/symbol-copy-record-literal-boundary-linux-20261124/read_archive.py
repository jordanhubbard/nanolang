"""I verify the original copy qualifier refusal and its independent controls."""
from pathlib import Path
import sys,json,hashlib,subprocess,re
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'b1e89cf7e662a215661a6bfc50fc4a692acfa6106cd3973710504fe34391001a');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('component/result.json');quick=js('component/quick-result.json');cohort=js('component/cohort-result.json');local=js('component/localization-result.json');vm=js('vm-diagnostics/result.json');w=js('witness/result.json');g=js('observer/result.json')
assert m['source_commit']==s['source_base']=='0ba525350d266bbed3d8f661db68d2fe5d87d52b' and m['overlay_source_commit']=='6e578585bb2eb6cbffc26db63bbaa1f19ec4dede'
assert m['diagnostic_only'] and not any(m[n] for n in ['product_acceptance','historical_cause','new_bootstrap','whole_release','first_cohort_native_acceptance','literal_native_acceptance'])
assert m['overlay_files']==['src_nano/typecheck.nano'] and s['working_overlay'] and s['all_guards_match']
assert all(q['terminal'] and q['all_guards_match'] for q in [quick,cohort,local,vm,w,g])
def guard(q,label,absolute_roots):
 for group in ['source','images','tools','hosts']:
  if group+'_before' not in q:continue
  assert q[group+'_before']==q[group+'_after']
  for name,h in q[group+'_before'].items():assert rows[('component/source/'+name) if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
 if 'inputs_before' in q:
  assert q['inputs_before']==q['inputs_after']
  for name,h in q['inputs_before'].items():
   if not name.startswith('/'):
    key=('component/components/'+name) if label=='quick' else 'component/'+name
   else:
    choices=[(prefix,root) for prefix,root in absolute_roots.items() if name.startswith(root+'/')];assert len(choices)==1,(label,name);prefix,root=choices[0];key=prefix+'/'+str(Path(name).relative_to(root))
   assert rows[key]['sha256']==h,(label,key)
 for command in q.get('commands',[]):
  assert command['timeout'] in [60,600]
  assert command['exit'] in [0,1] and not command.get('timed_out')
roots=m['roots']
for q,label in [(quick,'quick'),(cohort,'cohort'),(local,'local'),(vm,'vm'),(w,'witness')]:guard(q,label,roots)
assert quick['accepted'] and quick['ordinary_seconds']==10 and all(c['exit']==0 for c in quick['commands'])
assert quick['product']['sha256']==rows['component/products/scope-quick.nvm']['sha256']=='91081eb7411ff21ce742316f01db7a74cf724f16346ec63db40ac90b992c2917' and quick['product']['bytes']==790636
assert data('component/quick-vm.stdout')==b'SCOPE-COPY-OK\n'
assert len(cohort['commands'])==3 and [c['exit'] for c in cohort['commands']]==[0,0,1] and not cohort.get('accepted')
reason=b'I found conflicting record-array literal field representations'
assert reason in data('component/c-producer-translate.stderr') and 'component/products/scope-candidate.c' not in rows
assert local['accepted_localization'] and local['both_c_native_full_diagnostics_match'] and local['baseline_native_translation_exit']==1 and local['baseline_prior_output_preserved']
assert local['baseline_module']['sha256']==rows['component/products/scope-baseline.nvm']['sha256']=='a2b9286ca6d9635298e6323d1f610368fedb0068dca706ac3e490c8969e5bd14' and local['baseline_module']['bytes']==790356
assert data('component/products/scope-baseline-translated.c')==b'prior native C output\n' and reason in data('component/local-baseline-translate.stderr')
assert data('component/local-baseline-vm.stdout')==data('component/local-copy-baseline.stdout')==data('component/local-copy-candidate.stdout')==b'SCOPE-COPY-OK\n'
assert vm['accepted_observation'] and vm['all24_vm_pairs_equal'] and vm['all24_vm_native_equal'] and len(local['cases'])==len(vm['cases'])==24
assert all(c['exit']==0 and c['timeout']==60 for c in vm['commands']) and len(vm['commands'])==48
cases=js('component/diagnostics/cases.json')['cases'];assert len(cases)==24
for index,case in enumerate(cases):
 expected=data('component/diagnostics/case-%02d.c-native-expected'%index)
 assert expected.startswith(b'SCOPE-COPY-OK\nTYPECHECK\n')
 for label in ['baseline','candidate']:
  assert data('component/local-diag-%02d-'%index+label+'.stdout')==expected
  assert data('vm-diagnostics/case-%02d-'%index+label+'.stdout')==expected
 assert local['cases'][index]['report_sha256']==vm['cases'][index]['vm_report_sha256']==sha(expected)
 assert rows['component/diagnostics/'+case['path']]['sha256']==local['cases'][index]['source_sha256']==vm['cases'][index]['source_sha256']
launcher=js('vm-diagnostics/launcher-boundary.json');assert launcher['observed_pid_absent'] and launcher['first_receipt_absent'] and not launcher['test_execution_established'] and not launcher['historical_cause']
assert not launcher['first_stdout_bytes'] and not launcher['first_stderr_bytes'] and not data('vm-first/controller.stdout') and not data('vm-first/controller.stderr') and 'vm-first/result.json' not in rows
assert data('vm-first/run.py')==data('vm-diagnostics/run.py')
assert w['accepted_observation'] and w['ordinary_seconds']==10 and w['cases']['literal']['native_translation_exit']==1 and w['cases']['literal']['prior_output_preserved'] and w['cases']['literal']['c_native_pass']
assert w['cases']['append']['native_translation_exit']==0 and w['cases']['append']['c_native_pass'] and w['cases']['append']['aot_pass']
assert w['cases']['literal']['module_sha256']==rows['witness/literal.nvm']['sha256']=='61def753dae7059f528f192dbb74706fd3269ebede142b5a90275bc10f4d35f6'
assert w['cases']['append']['module_sha256']==rows['witness/append.nvm']['sha256']=='4a9914feef4d4007ecb9ccd54c889d8942c65baa5029ff3219adc4018178cc8d'
assert data('witness/literal.c')==b'prior C output\n' and reason in data('witness/literal-translate.stderr')
for name in ['literal-vm','literal-c-native-run','append-vm','append-c-native-run','append-aot-run']:assert data('witness/'+name+'.stdout')==b'SAME-NSTYPE-OK\n'
assert all(c['exit']==0 for c in w['commands'] if c['label']!='literal-translate')
aot=next(c['argv'] for c in w['commands'] if c['label']=='append-aot-build');assert all(flag in aot for flag in ['-fsanitize=address,undefined','-fno-sanitize-recover=all','-O1','-std=c11'])
literal=data('witness/literal.nano');append=data('witness/append.nano')
assert literal.replace(b'let values: array<NSType> = [(type_int), (type_array_of (type_bool))]',b'let mut values: array<NSType> = []\n    set values (array_push values (type_int))\n    set values (array_push values (type_array_of (type_bool)))')==append
assert g['accepted_observation'] and g['diagnostic_only'] and g['timeout']==60 and g['exit']==0 and g['seconds']<60
assert g['tool_sha256']==g['tool_after']==rows['component/source/bin/nvm2c']['sha256'] and g['module_sha256']==g['module_after']==local['baseline_module']['sha256']
assert g['debugger_sha256']==g['debugger_after']==rows['external/usr/bin/gdb']['sha256'] and g['script_sha256']==rows['observer/observe.gdb']['sha256']
assert g['prior_sha256']==g['output_after']==rows['observer/prior.c']['sha256'] and data('observer/prior.c')==b'prior native C output\n'
log=data('observer/gdb.stdout')
for marker in [b'idx = 566',b'start = 133',b'ins.opcode = 88',b'count = 2',b'f = 3',b'array.rec_k[f] = 8',b'value.rec_k[f] = 1',b'(process ',b') killed]']:assert marker in log,marker
functions=[line for line in data('component/baseline-dump.stdout').decode().splitlines() if line.startswith('.function ')];assert functions[566].startswith('.function scope_check_copy ') and sum(line.startswith('.function env_copy_symbols ') for line in functions)==1
candidate=data('component/components/candidate.nano');baseline=data('component/components/baseline.nano');path=str(Path(roots['component'])/'components/baseline-typecheck.nano')
assert baseline.replace(('"'+path+'"').encode(),b'"src_nano/typecheck.nano"',1)==candidate
old=data('component/components/baseline-typecheck.nano');new=data('component/source/src_nano/typecheck.nano')
header=b'fn env_copy_symbols(symbols: array<Symbol>) -> array<Symbol> {\n';stop=b'\nshadow env_copy_symbols {'
old_fn=old[old.index(header):old.index(stop,old.index(header))];new_fn=new[new.index(header):new.index(stop,new.index(header))]
added=b'    let count: int = (array_length symbols)\n    if (<= count 4294967295) {\n        return (array_slice symbols 0 count)\n    }\n    # I retain my original loop beyond the native slice endpoint range.\n'
assert new_fn==old_fn.replace(header,header+added,1)
assert old_fn.replace(b'fn env_copy_symbols',b'fn scope_manual_copy') in candidate
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git_hash(commit,name):
 proc.stdin.write((commit+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',name;raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return sha(raw)
for name,h in s['source_before'].items():
 assert git_hash(m['overlay_source_commit'] if name in m['overlay_files'] else m['source_commit'],name)==h,name
 if name.endswith(('.c','.h','.inc')):assert git_hash(m['compiled_tool_source_commit'],name)==h,name
assert git_hash(m['source_commit'],'src_nano/typecheck.nano')==sha(old) and git_hash(m['overlay_source_commit'],'tests/typecheck_scope_copy.nano')==sha(candidate)
proc.stdin.close();assert proc.wait()==0
parent=base.parent/'lexer-byte-snapshot-development-linux-20261124';pm,pr,pdata,pjs=read_archive(parent,'b7caa84b508dec5cc7cc160cd8d3acdc1c855946e9f643319c1fa08c359460f8');origin=pjs('development/result.json')
for name,h in s['images_before'].items():assert h==pr['development/source/'+name]['sha256']
assert s['producer_sha256']==origin['producer_sha256']==rows['component/source/bin/nanoc_emitter_phase']['sha256']
print('PASS %d records: original native qualifier refusal; original/current C-native+VM24 diagnostic records and14 copy-size controls; exact GDB literal field3 VALUE/STR, reduced literal refusal/prior preservation, unchanged append neighbor VM/C-native/ASan/UBSan PASS. Detached launcher absence retained; source/tool/host guards stable. No full five-route, factory, Stage2/paired/pure-translator/release acceptance.'%len(rows))
