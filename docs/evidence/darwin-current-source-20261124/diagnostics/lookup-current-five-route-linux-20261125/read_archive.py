"""I verify the original complete lookup gate with an actual bootstrapped producer."""
from pathlib import Path
import hashlib,json,shlex,subprocess,sys
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base.parent))
from read_chunks import read_archive
m,rows,data,js=read_archive(base,'e0d1e184ad49963a42e8252b907ab1db78e4020619e38492a31f3f73ab8cfbcf')
sha=lambda b:hashlib.sha256(b).hexdigest()
s=js('component/result.json');q=js('component/cohort-result.json');a=js('component/admission-result.json');hosts=js('component/runtime-hosts-after.json')
pin='95a4062efebe0e97d8cde271117eac1deab21595'
producer='78228e7dbd4485d1b243088a2d25ac5578816e08e1a48d4e0792cd47bf086739'
assert m['source_commit']==s['source_base']==pin and m['component_only'] and m['full_five_routes'] and m['published_module_hosts_guarded']
assert not any(m[k] for k in ['new_bootstrap','new_compiler_image','complete_runtime_sanitizers','whole_release','historical_cause','performance_claim'])
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['overlays'] and not s['stage_alias']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3
assert a['output_directories']==dict(products=True,tmp=True)
assert all(v['cpu']>=1 and v['share']>=.75 for v in s['cpu_rounds'])
assert q['terminal'] and q['accepted_component'] and q['all_guards_match'] and q['ordinary_seconds']==10
assert q['permanent_test']==dict(tests=1,failures=0,errors=0,skips=0,passed=True)
assert len(q['commands'])==132 and q['original_stock_build_commands']==7
assert [v['label'] for v in q['routes']]==['c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native']
for command in q['commands']:
 assert command['exit']==0 and not command.get('timed_out') and command['timeout']==(600 if command['number']<=7 else 60)
 assert 'NANO_SHADOW_TIMEOUT_SECONDS' not in command['argv'] and '--no-tests' not in command['argv']
for n in [4,7]:assert all(flag in q['commands'][n-1]['argv'] for flag in ['-std=c11','-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all'])
for n in range(8,13):assert data('component/command-%d.stdout'%n)==b'LOOKUP-PROJECTION-OK\n'
corpus=js('component/source/tests/typecheck_scope_copy_cases.json')['cases'];expected=js('component/source/tests/nanoisa_lookup_projection_cases.json')
assert len(corpus)==len(expected)==24
for i,(case,wanted) in enumerate(zip(corpus,expected)):
 source=bytes.fromhex(case['source_hex']);assert sha(source)==wanted['source_sha256']
 assert data('component/test-inputs/case-%d.nano'%i)==source
 for n in range(13+i*5,18+i*5):assert sha(data('component/command-%d.stdout'%n))==wanted['report_sha256'],(i,n)
assert b'Ran 1 test in ' in data('component/permanent-test.stderr') and data('component/permanent-test.stderr').rstrip().endswith(b'OK')
for group in ['source','images','tools','hosts']:
 assert q[group+'_before']==q[group+'_after']==s[group+'_before']==s[group+'_after']
 for name,h in q[group+'_before'].items():
  key='component/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')
  assert rows[key]['sha256']==h
assert q['source_producer_sha256']==q['producer_after']==s['producer_sha256']==rows['component/source/bin/nanoc_emitter_phase']['sha256']==producer
assert q['source_producer_kind']==s['producer_kind']=='original-Make-bootstrap-own-Stage2' and s['producer_source_pin']==pin
for name,v in q['products'].items():assert rows['component/products/'+name]['sha256']==v['sha256'] and rows['component/products/'+name]['bytes']==v['bytes']
assert hosts['terminal'] and hosts['all_recorded_guards_match'] and hosts['cohort_receipt_sha256']==rows['component/cohort-result.json']['sha256']
for label in ['decoder','source-decoder']:
 before=js('component/'+label+'-runtime-hosts-before.json');after=hosts['observations'][label]
 assert before['terminal'] and before['read_only'] and not before['new_product_execution'] and before['cohort_command_count']<8 and before['exit']==0 and not before['logical_namespaces']
 assert before['module_sha256']==before['module_after']==after['module_sha256_after']==rows['component/products/'+label+'.nvm']['sha256']
 assert before['libraries']==after['libraries_after'] and after['before_receipt_sha256']==rows['component/'+label+'-runtime-hosts-before.json']['sha256']
 assert before['tool_sha256_before']==before['tool_sha256_after']==s['images_before']['bin/nanoisa']
 assert before['bindings']==[shlex.split(line) for line in data('component/'+label+'-runtime-host-dump.stdout').decode().splitlines() if line.startswith('.import ')]
 for name,h in before['libraries'].items():assert rows['external/'+name.lstrip('/')]['sha256']==h
pm,pr,pd,pj=read_archive(base.parent/'lookup-kernel-cold-bootstrap-linux-20261125','667a97fd742fd62558ed12eefa3578906acc55f8364131bd0b1f0171feb883d1')
b=pj('bootstrap/bootstrap-result.json');installed=pj('bootstrap/source/bin/nanoc_bootstrap.json')
assert b['terminal'] and b['original_make_pass'] and b['exit']==0 and installed['phase']=='stage2' and installed['raw_stage1_stage2_equal']
assert s['source_before']==b['source_after'] and s['hosts_before']==installed['hosts'] and installed['artifacts']['nanoc_stage2']==producer
assert s['bootstrap_result_sha256']==pr['bootstrap/bootstrap-result.json']['sha256'] and s['bootstrap_receipt_sha256']==pr['bootstrap/source/bin/nanoc_bootstrap.json']['sha256']
# My permanent method and stock seven-command route builder remain exact Git inputs.
assert b'len(products), 5 if routes.source_compiler_command() else 3' in data('component/source/tests/test_nanoisa_lookup_projection.py')
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob'
 raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: exact95a/actual78228 origin, all132 original commands, all24 complete reports through five routes, both generated sanitizer natives, directory admission and stable published-module/source/image/tool/provider guards. No later-source/factory/paired/pure-translator/release or cause/gain acceptance.'%len(rows))
