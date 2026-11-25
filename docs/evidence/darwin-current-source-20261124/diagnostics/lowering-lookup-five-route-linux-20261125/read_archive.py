"""I retain the failed source-produced build without granting five-route acceptance."""
from pathlib import Path
import sys,json,hashlib,subprocess,struct
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'f152e7370149e34cdbae8863f053c57ef28ebb2a239b47a103474c62e305435f');sha=lambda raw:hashlib.sha256(raw).hexdigest();s=js('qualification/result.json');q=js('qualification/cohort-result.json');a=js('qualification/admission-result.json')
assert m['component_only'] and m['source_producer_build_timeout'] and m['permanent_methods_run']==0 and not any(m[k] for k in ['full_five_routes','new_bootstrap','whole_release','performance_claim'])
assert m['source_commit']==s['source_base']=='71721fcfaf4424e12b46844915626ef3f3249361'
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and not s['working_overlay'] and not s['overlays'] and not s['new_bootstrap'] and not s['stage_alias'] and len(s['source_before'])==3910
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['filesystem']!='nfs' and a['free_bytes']>15*1024**3 and all(x['cpu']>=1 and x['share']>=.75 for x in s['cpu_rounds'])
assert q['terminal'] and q['all_guards_match'] and q['ordinary_seconds']==10 and len(q['commands'])==5 and 'permanent_test' not in q and 'accepted_component' not in q and 'routes' not in q
for row in q['commands'][:4]:assert row['exit']==0 and not row.get('timed_out') and row['timeout']==600
last=q['commands'][-1];assert last['number']==5 and last['timeout']==600 and last['timed_out'] and last['exit']==-9 and 600<=last['seconds']<610
assert last['argv']==[s['producer'],str(Path(m['roots']['qualification'])/'source/tests/nanoisa_lookup_projection.nano'),'--emit-nvm','-o',str(Path(m['roots']['qualification'])/'products/source-decoder.nvm')]
assert data('qualification/command-5.stdout')==data('qualification/command-5.stderr')==b'' and b'AssertionError' in data('qualification/cohort.stderr')
assert not any(name.startswith('qualification/products/source-decoder') or name=='qualification/permanent-test.stderr' for name in rows)
assert s['producer_kind']==q['source_producer_kind']=='self-host-produced-development' and s['producer_origin_evidence']==q['source_producer_origin']=='76f1172b2b90aa8f186e46b9a602b75c165e041a'
assert s['producer_sha256']==q['source_producer_sha256']==q['producer_after']==rows['qualification/source/bin/nanoc_lookup_producer']['sha256']=='385db586c833e94a5a7d02eefa6289a5020bf3e46ecb75936d466f45f712fb4c'
for group in ['source','images','tools','hosts']:
 assert s[group+'_before']==s[group+'_after']==q[group+'_before']==q[group+'_after']
 for name,h in s[group+'_after'].items():assert rows['qualification/source/'+name if group in ['source','images'] else 'external/'+name.lstrip('/')]['sha256']==h
for name,h in s['producer_origin_receipts'].items():assert rows['producer-origin/'+name]['sha256']==h
origin=js('producer-origin/development-native-result.json');native36=js('producer-origin/native36-result.json');factory=js('producer-origin/factory-result.json');assert origin['terminal'] and origin['development_native_accepted'] and origin['all_guards_match'] and origin['source_base']==s['producer_source_pin']=='00025ff76979b4009b84f10f8cb8f07492c62210' and native36['accepted_component'] and factory['timed_out'] and not factory['product']
assert s['hosts_before']==origin['hosts_after']
before=js('qualification/decoder-runtime-hosts-before.json');after=js('observation/decoder-hosts-after.json');assert before['terminal'] and before['exit']==0 and before['cohort_command_count']==4 and after['terminal'] and after['libraries']==before['libraries'] and after['module_sha256']==before['module_sha256']==rows['qualification/products/decoder.nvm']['sha256']
assert after['before_receipt_sha256']==rows['qualification/decoder-runtime-hosts-before.json']['sha256'] and after['after_failed_cohort_sha256']==rows['qualification/cohort-result.json']['sha256']
for name,h in after['libraries'].items():assert rows['external/'+name.lstrip('/')]['sha256']==h
for name in ['decoder-native','decoder-aot']:
 raw=data('qualification/products/'+name);assert raw[:4]==b'\x7fELF' and raw[4:6]==bytes([2,1]) and struct.unpack_from('<H',raw,18)[0]==183
assert data('qualification/products/decoder.nvm')[:4]==b'NVM\x02'
assert '-fsanitize=address,undefined' in q['commands'][3]['argv'] and '-fno-sanitize-recover=all' in q['commands'][3]['argv']
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip();proc=subprocess.Popen(['git','cat-file','--batch'],cwd=repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_before'].items():
 proc.stdin.write((m['source_commit']+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob';raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n' and sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS %d records: exact3910 inputs, actual385db origin, four successful builds including instrumented C product; source-produced build times out original600/default10, exit-9/no module, zero permanent methods. Guards/one module closure stable; no functional sanitizer, full-five, bootstrap, paired, cutover, release or gain acceptance.'%len(rows))
