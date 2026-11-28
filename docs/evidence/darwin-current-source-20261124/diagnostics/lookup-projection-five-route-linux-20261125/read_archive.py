"""I verify all five lookup routes and independently guarded emitted-module bindings."""
from pathlib import Path
import sys,json,hashlib,subprocess
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'ffa41e7406625c7dbbe409dd1dfc5eadf6e4aef191b64ae61caae49f827040bf');sha=lambda raw:hashlib.sha256(raw).hexdigest()
s=js('component/result.json');q=js('component/cohort-result.json');a=js('component/admission-result.json');hosts=js('component/runtime-hosts-after.json');ob=js('component/projected-routes-materialization.json')
assert m['source_commit']==s['source_base']=='8806139ca8bf3f60a25d88d241f67747c6ef9e8a'
assert m['component_only'] and m['full_five_routes'] and m['published_module_hosts_guarded'] and not any(m[k] for k in ['new_bootstrap','new_compiler_image','whole_release','historical_cause','performance_claim'])
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
assert hosts['terminal'] and hosts['all_recorded_guards_match'] and hosts['cohort_receipt_sha256']==rows['component/cohort-result.json']['sha256']
for label in ['decoder','source-decoder']:
 before=js('component/'+label+'-runtime-hosts-before.json');after=hosts['observations'][label]
 assert before['terminal'] and before['read_only'] and not before['new_product_execution'] and before['cohort_command_count']<8 and before['exit']==0 and not before['logical_namespaces']
 assert before['module_sha256']==before['module_after']==after['module_sha256_after']==rows['component/products/'+label+'.nvm']['sha256']
 assert before['libraries']==after['libraries_after'] and after['before_receipt_sha256']==rows['component/'+label+'-runtime-hosts-before.json']['sha256']
 assert before['tool_sha256_before']==before['tool_sha256_after']==s['images_before']['bin/nanoisa']
 import shlex
 assert before['bindings']==[shlex.split(line) for line in data('component/'+label+'-runtime-host-dump.stdout').decode().splitlines() if line.startswith('.import ')]
 for path,h in before['libraries'].items():assert rows[('component/source/'+str(Path(path).relative_to(Path(m['roots']['component'])/'source'))) if path.startswith(str(Path(m['roots']['component'])/'source')+'/') else 'external/'+path.lstrip('/')]['sha256']==h
 facts=ob['functions'][label];raw=data('component/products/'+label+'.c');text=raw.decode();import re
 matches=list(re.finditer(r'^static int64_t (\w*_env_lookup)\([^\n]*\) \{$',text,re.M));assert len(matches)==1;start=matches[0].start();body=text[start:text.index('\n}\n',start)+3]
 assert sha(raw)==facts['file_sha256'] and sha(body.encode())==facts['excerpt_sha256'] and body.encode()==data('component/'+label+'-env_lookup.c')
 assert not re.search(r'nrec_t \*rl = calloc\(',body) and facts['record_locals']==facts['local_record_roots']==0
 assert facts['record_temporaries']==int(re.search(r'nrec_t \*r = (\d+) \? calloc',body).group(1))==1
 assert facts['array_record_gets']==body.count('nrarr_get(')==1 and facts['print_calls']==body.count('fputs(')==1 and facts['flush_calls']==body.count('fflush(')==1
assert len(js('component/decoder-runtime-hosts-before.json')['libraries'])==1 and not js('component/source-decoder-runtime-hosts-before.json')['libraries']
qm,qr,qd,qj=read_archive(base.parent/'lookup-projection-quick-linux-20261125','8daef2c428d6701398da9829b3d5fb016cc6b3e2e0c8ef9676e050a5ed4b59b0');assert qj('quick/quick-result.json')['accepted_c_native_vm_boundary']
assert data('component/source/src_nano/typecheck.nano')==qd('quick/source/src_nano/typecheck.nano') and data('component/source/tests/typecheck_scope_copy.nano')==qd('quick/source/tests/typecheck_scope_copy.nano')
assert data('component/source/tests/typecheck_scope_copy_cases.json')==qd('quick/source/tests/typecheck_scope_copy_cases.json')
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def git_data(commit,name):
 proc.stdin.write((commit+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob',name;raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';return raw
for name,h in s['source_before'].items():assert sha(git_data(m['source_commit'],name))==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: exact lookup source/full132 stock commands, original14 sizes and24 complete reports across five routes; both generated sanitizer natives pass. Both emitted modules/native-library bindings are guarded before functional commands and after; both retained C routes use one temporary/no retained Symbol local. No new compiler/factory/bootstrap/paired/full-translator/release or gain acceptance.'%len(rows))
