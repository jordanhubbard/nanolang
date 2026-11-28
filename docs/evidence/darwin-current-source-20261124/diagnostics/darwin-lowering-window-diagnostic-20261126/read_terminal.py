"""I verify a reversible observation overlay without accepting a compiler."""
from pathlib import Path
import argparse,json,gzip,hashlib,subprocess,sys,re,struct,zlib
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path);p.add_argument('--parent',type=Path);p.add_argument('--failure',type=Path);a=p.parse_args();base=Path(__file__).resolve().parent;parent=a.parent or base.parent/'darwin-stage1-phase-diagnostic-20261126';failure=a.failure or base.parent/'darwin-stage1-generation-timeout-20261126';sha=lambda b:hashlib.sha256(b).hexdigest();m=json.loads((base/'manifest.json').read_bytes());rows={r['path']:r for r in m['files']};assert len(rows)==len(m['files'])
supplement=json.loads((base/'supplement.json').read_bytes());assert supplement['original_manifest_sha256']==sha((base/'manifest.json').read_bytes()) and supplement['post_terminal_read_only_analysis']
assert [r['path'] for r in supplement['files']]==['metadata/generated_schema.h','metadata/opcode-observation.json'];rows.update({r['path']:r for r in supplement['files']})
def raw(n):
 r=rows[n];path=Path(r['stored']);assert not path.is_absolute() and '..' not in path.parts;b=(base/path).read_bytes();assert len(b)==r['stored_bytes'] and sha(b)==r['stored_sha256'];b=gzip.decompress(b);assert len(b)==r['bytes'] and sha(b)==r['sha256'];return b
for n in rows:raw(n)
load=lambda n:json.loads(raw(n));q=load('controller/result.json');request=load('controller/request.json');audit=load('terminal-audit.json');frozen=load('frozen-parent-audit.json')
assert q['terminal'] and q['diagnostic_only'] and not q['accepted'] and not q['whole_release'] and m['observation_only'] and not m['compiler_accepted'] and not m['whole_release']
assert q['source_pin']==request['source_pin']==m['source_pin']=='d907f264d16d348b4589d3760bd213803e1c6f35'
assert request['parent_publication']==m['parent_publication']=='2504d8f9606d208370e05eb524c0e86531f464f5'
assert sha((parent/'manifest.json').read_bytes())==request['parent_manifest_sha256']==m['parent_manifest_sha256']
assert sha((parent/'read_terminal.py').read_bytes())==request['parent_reader_sha256']
assert sha((failure/'manifest.json').read_bytes())==request['failure_manifest_sha256']==m['failure_manifest_sha256']
assert sha((failure/'supplement.json').read_bytes())==request['failure_supplement_sha256']==m['failure_supplement_sha256']
cmd=[sys.executable,str(parent/'read_terminal.py'),'--parent',str(failure)]+(['--repo',str(a.repo)] if a.repo else []);checked=subprocess.run(cmd,capture_output=True,text=True);assert checked.returncode==0,checked.stderr;parent_result=json.loads(checked.stdout)
pm=json.loads((parent/'manifest.json').read_bytes());pr={r['path']:r for r in pm['files']};parentraw=lambda n:gzip.decompress((parent/pr[n]['stored']).read_bytes());pq=json.loads(parentraw('controller/result.json'))
assert parent_result['diagnostic_valid'] and not parent_result['compiler_accepted']
assert all(sha(raw('controller/'+n))==h for n,h in request['observer_sources'].items())
binding=load('controller/overlay-binding.json');old=raw('controller/baseline-codegen.nano').decode();observed=raw('controller/observed-codegen.nano').decode();name=request['overlay_path'];assert name==binding['path']=='src_nano/compiler/nanoisa_codegen.nano'
assert binding['source_pin']==q['source_pin'] and binding['offset_units']=='unicode-codepoints' and binding['target_function']=='nisa_emit_module_indexed_body'
assert sha(old.encode())==binding['baseline_sha256']==pq['source_after'][name] and sha(observed.encode())==binding['observed_sha256']==q['source_after'][name]
start,end=binding['start'],binding['end'];body=old[start:end]
for op in binding['operations']:
 assert body.count(op['before'])==1 and op['after'].count(op['before'])==1;body=body.replace(op['before'],op['after'],1)
assert len(binding['operations'])==12 and binding['added_shadow'].count('\nshadow ')==1 and binding['added_shadow'].startswith('\nshadow nisa_emit_module_indexed_body {')
assert observed==old[:start]+body+binding['added_shadow']+old[end:]
for op in reversed(binding['operations']):assert body.count(op['after'])==1;body=body.replace(op['after'],op['before'],1)
assert body==old[start:end]
assert q['baseline_sources']==q['parent_sources_before']==q['parent_source_after']==pq['source_after']
expected=dict(q['baseline_sources']);expected[name]=binding['observed_sha256'];assert q['source_before']==q['source_after']==expected
assert q['images_before']==q['images_after']==q['parent_images_before']==q['parent_images_after']==pq['images_after'];assert q['tools_before']==q['tools_after']==pq['tools_after']
assert q['parent_result_before']==q['parent_result_after']==sha(parentraw('controller/result.json'))
assert q['guards_match'] and not q['guard_errors'] and not q['remaining_owned_processes'] and q['reservation_released']
assert frozen['match'] and frozen['expected']==frozen['actual'];assert audit['source_match'] and audit['images_match'] and audit['tools_match'] and audit['hosts_match'] and not audit['remaining_owned_processes'] and not audit['reservation_exists']
probe=audit['controller_probe'];assert (probe['exit']==1 and not probe['stdout']) or (probe['exit']==0 and probe['stdout'].strip() and not probe['stdout'].strip().startswith(q['process_start']))
assert (request['seed_seconds'],request['verify_seconds'],request['compiler_shadow_seconds'],request['ordinary_seconds'],request['target_entry_seconds'],request['target_window_seconds'],request['original_qualification_stage_seconds'])==(600,60,30,10,600,120,1800)
for admission in q['admissions']:assert admission['passed'] and admission['filesystem']=='apfs' and admission['free_bytes']>15*1024**3 and not admission['other_nanolang_jobs'] and len(admission['cpu'])==2 and all(c['share']>=.75 for c in admission['cpu'])
for c in q['commands']:
 for stream in ['stdout','stderr']:
  if stream+'_sha256' in c:assert sha(raw('controller/'+c['label']+'.'+stream))==c[stream+'_sha256']
assert [c['label'] for c in q['commands']]==['parent-reader','seed-prefix','lowering-window'] and all(c['exit']==0 for c in q['commands'][:2])
prefix=load('controller/prefix-result.json');r=prefix['manifest'];assert prefix['prepared_prefix_only'] and prefix['stopped_before_stage1'] and not prefix['bootstrap_accepted'] and not prefix['whole_release'];assert prefix['budgets']==[{'label':'seed','seconds':600},{'label':'seed-verify','seconds':60}]
assert r['phase']=='building-stage1' and r['stage_timeout_seconds']==600 and r['shadow_timeout_seconds']==30 and [v['label'] for v in r['steps']]==['seed','seed-verify'] and all(v['exit_code']==0 for v in r['steps'])
for step in r['steps']:assert sha(raw('bootstrap/'+Path(r['work']).name+'/'+Path(step['log']).name))==step['log_sha256']
seed=raw('products/bin/nanoc_seed.nvm');assert seed.startswith(b'NVM\x02') and q['new_seed_before']==q['new_seed_after']==r['artifacts']['nanoc_seed.nvm']==sha(seed) and set(r['artifacts'])=={'nanoc_seed.nvm'}
def seed_sections(b):
 h=struct.unpack_from('<4sHHIQIIIII',b);assert h[0]==b'NVM\x02' and h[1:3]==(1,2) and h[4]==len(b) and h[5]==40 and not h[8] and zlib.crc32(b[40:])==h[9];out={}
 for i in range(h[6]):
  k,flags,offset,size=struct.unpack_from('<IIQQ',b,40+24*i);assert 1<=k<=14 and not flags and k not in out and offset>=40+24*h[6] and size<=len(b)-offset;out[k]=b[offset:offset+size]
 return h,out
def opcode_histogram(blob,header):
 import re,struct,collections
 text=header.decode().split('static const NanoisaSchemaOpcode nanoisa_schema_opcodes[] = {',1)[1].split('\n};',1)[0]
 rows=re.findall(r'^\s*\{"([A-Z0-9_]+)", (0x[0-9a-f]+), (\d+), \{([^}]+)\}, -?\d+, -?\d+\},$',text,re.M);assert len(rows)==191
 widths={'NONE':0,'U8':1,'U16':2,'U32':4,'I32':4,'I64':8,'F64':8};ops={}
 for name,number,count,operand_text in rows:
  operands=[v.strip().removeprefix('OPERAND_') for v in operand_text.split(',')];n=int(count);assert all(v=='NONE' for v in operands[n:]) and all(v!='NONE' for v in operands[:n]);number=int(number,0);assert number not in ops;ops[number]=(name,1+sum(widths[v] for v in operands[:n]))
 sections={}
 for i in range(struct.unpack_from('<I',blob,24)[0]):kind,flags,at,n=struct.unpack_from('<IIQQ',blob,40+24*i);sections[kind]=blob[at:at+n]
 functions,bytecode=sections[5],sections[6];hist=collections.Counter()
 for i in range(struct.unpack_from('<I',functions)[0]):
  at,n=struct.unpack_from('<QQ',functions,4+32*i+8);end=at+n;assert end<=len(bytecode)
  while at<end:
   name,width=ops[bytecode[at]];hist[name]+=1;at+=width
  assert at==end
 return dict(sorted(hist.items()))

old_header,old_sections=seed_sections(parentraw('products/bin/nanoc_seed.nvm'));header,sections=seed_sections(seed)
assert header[3]==old_header[3]==2 and struct.unpack_from('<I',sections[5])[0]==struct.unpack_from('<I',old_sections[5])[0]==1350
assert all(sections[n]==old_sections[n] for n in [3,4,7,9])
generated=raw('metadata/generated_schema.h');assert sha(generated)==q['baseline_sources']['src/nanoisa/generated_schema.h']
x=opcode_histogram(parentraw('products/bin/nanoc_seed.nvm'),generated);y=opcode_histogram(seed,generated);report=load('metadata/opcode-observation.json');assert report['old_histogram']==x and report['new_histogram']==y and report['old_opcode_kinds']==report['new_opcode_kinds']==len(x)==len(y)==57 and report['introduced']==report['removed']==[] and set(x)==set(y)
assert report['delta']=={k:y.get(k,0)-x.get(k,0) for k in sorted(set(x)|set(y)) if x.get(k)!=y.get(k)}
for label in re.findall(r'native_phase_marker "(lower-[a-z-]+)"',observed[start:observed.index(binding['added_shadow'],start)]):assert label.encode() in seed,label
assert q['new_hosts_before']==q['new_hosts_after']==r['hosts'] and len(r['hosts'])==6
for n,h in r['hosts'].items():assert sha(raw('hosts/'+str(Path(n).relative_to(Path(q['root'])/'source'))))==h
assert q['working_sources_before']==q['working_sources_after']==r['sources'];extra=set(r['sources'])-set(q['source_after']);assert extra=={'modules/index.json'}
for n,h in r['sources'].items():assert (sha(raw('inputs/'+n)) if n in extra else q['source_after'][n])==h
assert q['phase_record_complete'] and q['diagnostic_valid'] and not q.get('exception') and not q['unexpected_native_compiler'] and q['phases']==['seed-prefix','lowering-window']
events=[json.loads(line) for line in raw('controller/phases.jsonl').splitlines()];assert events==q['phase_events'] and all(x['monotonic']<=y['monotonic'] for x,y in zip(events,events[1:]));targets=[e for e in events if e['phase']=='program-lowering-start'];assert targets
last=q['commands'][-1];assert last['target_entry_seconds']==600 and last['target_window_seconds']==120 and last['target_started_monotonic']==targets[0]['monotonic'];assert last['stop_reason'] in ['target-return-marker','target-window-deadline','vm-exited']
assert 0<=last['target_started_monotonic']-last['started_monotonic']<=600
if last['stop_reason']=='target-window-deadline':assert last['deadline_monotonic']==last['target_started_monotonic']+120 and last['stop_monotonic']>=last['deadline_monotonic'] and last['observer_stopped_vm']
print(json.dumps(dict(verified=True,records=len(rows),parent_records=parent_result['records'],failure_records=parent_result['parent_records'],source_blobs_verified=parent_result['source_blobs_verified'],source_reversal=True,diagnostic_valid=True,stop_reason=last['stop_reason'],exit=last['exit'],seconds=last['seconds'],lowering_events=[dict(phase=e['phase'],seconds=e['monotonic']-last['target_started_monotonic']) for e in events if e['phase'].startswith('lower-')],compiler_accepted=False,whole_release=False,current_process_liveness_checked=False)))
