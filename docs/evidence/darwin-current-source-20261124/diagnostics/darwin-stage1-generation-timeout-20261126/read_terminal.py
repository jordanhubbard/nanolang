"""I verify retained terminal bytes and distinguish a factory result from release acceptance."""
from pathlib import Path
import argparse,gzip,hashlib,json,re,subprocess
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path);args=parser.parse_args();base=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest();manifest=json.loads((base/'manifest.json').read_bytes());records={r['path']:r for r in manifest['files']};assert len(records)==len(manifest['files'])
supplement=json.loads((base/'supplement.json').read_bytes());assert supplement['original_manifest_sha256']==sha((base/'manifest.json').read_bytes()) and supplement['source_pin']==manifest['source_pin'] and supplement['generated_inputs_only']
assert [r['path'] for r in supplement['files']]==['inputs/modules/index.json']
records.update({r['path']:r for r in supplement['files']})
def raw(name):
 r=records[name];p=Path(r['stored']);assert not p.is_absolute() and '..' not in p.parts;packed=(base/p).read_bytes();assert len(packed)==r['stored_bytes'] and sha(packed)==r['stored_sha256'],name;b=gzip.decompress(packed);assert len(b)==r['bytes'] and sha(b)==r['sha256'],name;return b
for name in records:raw(name)
load=lambda name:json.loads(raw(name));q=load('controller/result.json');audit=load('terminal-audit.json')
assert q['terminal'] and q['source_pin']==manifest['source_pin']==audit['source_pin'] and not q['whole_release'] and not manifest['whole_release']
assert (q['ordinary_seconds'],q['stage_seconds'],q['compiler_shadow_seconds'],q['emitter_helper_seconds'])==(10,1800,30,300)
assert not q['c_product_source_edit']
for command in q['commands']:
 for stream in ['stdout','stderr']:
  if stream+'_sha256' in command:assert sha(raw('controller/'+command['label']+'.'+stream))==command[stream+'_sha256']
for entry in q['admissions']:
 assert entry['passed'] and entry['filesystem']=='apfs' and entry['free_bytes']>15*1024**3 and not entry['other_nanolang_jobs']
 assert len(entry['cpu'])==2 and all(p['share']>=.75 for p in entry['cpu'])
assert q['source_before']==q['source_after'] and q['source_stable']
assert q['images_before']==q['images_after'] and q['tools_before']==q['tools_after'] and q['host_libraries_before']==q['host_libraries_after'] and q['guards_match']
assert audit['images_match'] and audit['tools_match'] and audit['hosts_match']
assert q['reservation_released'] and not q['remaining_owned_processes'] and not audit['remaining_owned_processes']
probe=audit['controller_probe'];assert (probe['exit']==1 and not probe['stdout']) or (probe['exit']==0 and probe['stdout'].strip() and not probe['stdout'].strip().startswith(q['process_start']))
source_checks=0
if args.repo:
 names=sorted(q['source_after']);pin=q['source_pin'];proc=subprocess.Popen(['git','cat-file','--batch'],cwd=args.repo,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
 for name in names:
  proc.stdin.write((pin+':'+name+'\n').encode());proc.stdin.flush();header=proc.stdout.readline().split();assert len(header)==3 and header[1]==b'blob',name;b=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n';assert sha(b)==q['source_after'][name],name;source_checks+=1
 proc.stdin.close();assert proc.wait()==0
accepted=False
if not q['accepted']:
 assert not manifest['controller_accepted'] and q['phases']==['cold-tools'] and q['exception']=="AssertionError('bootstrap')"
 assert q['commands'][-1]['label']=='bootstrap' and q['commands'][-1]['exit']==2 and q['commands'][-1]['timeout'] is None
 assert q['commands'][-1]['argv'][-1]=='bootstrap3'
 receipt=load('bootstrap/working-receipt.json');assert receipt['phase']=='building-stage1' and receipt['stage_timeout_seconds']==1800 and receipt['shadow_timeout_seconds']==30
 assert [v['label'] for v in receipt['steps']]==['seed','seed-verify','stage1']
 assert all(v['exit_code']==0 for v in receipt['steps'][:2]);last=receipt['steps'][-1]
 assert last['interrupted_or_timed_out'] and 1800<=last['elapsed_seconds']<1802 and 'exit_code' not in last
 assert b'timed out after 1800 seconds' in raw('controller/bootstrap.stderr')
 for step in receipt['steps'][:2]:assert sha(raw('bootstrap/'+Path(receipt['work']).name+'/'+Path(step['log']).name))==step['log_sha256']
 assert set(receipt['artifacts'])=={'nanoc_seed.nvm'} and len(receipt['hosts'])==6 and len(receipt['tools'])==5
 assert audit['receipt_artifacts_match'] and audit['receipt_sources_match'] and audit['receipt_tools_match'] and audit['receipt_hosts_match']
 extra={n for n in receipt['sources'] if n not in q['source_after']};assert extra=={'modules/index.json'}
 assert all(q['source_after'][n]==h for n,h in receipt['sources'].items() if n not in extra)
 for n in extra:assert sha(raw('inputs/'+n))==receipt['sources'][n]
 for name,h in receipt['artifacts'].items():assert sha(raw('products/bin/'+name))==h
 for name,h in receipt['tools'].items():assert sha(raw('products/bin/'+name))==h
 for name,h in receipt['hosts'].items():assert sha(raw('hosts/'+str(Path(name).relative_to(Path(q['source'])))))==h
 for name,h in q['images_after'].items():assert sha(raw('products/'+name))==h
 assert not audit['installed_link'] and not audit['raw_equal'] and not audit['unexpected_native_compiler'] and not audit['reservation_exists']
 assert set(audit['sentinels'])=={'.bootstrap0.built'} and all(not v['runs'] and not v['passed'] for v in audit['suites'].values())
 assert not any(c['label'] in ['native36','emitter12'] for c in q['commands'])
 assert raw('products/bin/nanoc_seed.nvm').startswith(b'NVM\x02')
 sample=load('controller/live-stage1-sample-observation.json');assert sample['exit']==0 and sample['read_only'] and sample['one_sample_seconds']==1 and sample['pid']==17638
 assert sha(raw('controller/live-stage1-sample.txt'))==sample['sample_sha256'] and 'Thu Nov 26 11:16:24 2026' in sample['process_before']
if q['accepted']:
 assert q['phases']==['cold-tools','bootstrap17','native36','emitter12'] and not q.get('exception')
 receipt=load('bootstrap/installed-receipt.json');assert receipt['phase']=='stage2' and receipt['version']==2 and receipt['stage_timeout_seconds']==1800 and receipt['shadow_timeout_seconds']==30
 assert len(receipt['steps'])==17 and all(s['exit_code']==0 for s in receipt['steps'])
 for step in receipt['steps']:
  name='bootstrap/'+Path(receipt['work']).name+'/'+Path(step['log']).name;assert sha(raw(name))==step['log_sha256']
 assert sha(raw('bootstrap/installed-receipt.json'))==q['bootstrap_receipt_sha256']
 assert raw('products/bin/nanoc_stage1.nvm')==raw('products/bin/nanoc_stage2.nvm')
 assert not audit['unexpected_native_compiler']
 assert receipt['raw_stage1_stage2_equal'] and audit['raw_equal'] and audit['raw_sha256']==q['raw_equality']['sha256'] and audit['raw_bytes']==q['raw_equality']['bytes']
 assert audit['receipt_artifacts_match'] and audit['receipt_sources_match'] and audit['receipt_tools_match'] and audit['receipt_hosts_match']
 for name,h in receipt['artifacts'].items():assert sha(raw('products/bin/'+name))==h
 for name,h in receipt['hosts'].items():assert sha(raw('hosts/'+str(Path(name).relative_to(Path(q['source'])))))==h
 for name,h in q['images_after'].items():assert sha(raw('products/'+name))==h
 assert audit['installed_link']=='nanoc_stage2' and not audit['git_status']
 assert all(audit['suites'][n]['passed'] for n in ['native36','emitter12'])
 for name,count in [('native36',36),('emitter12',12)]:
  b=raw('controller/'+name+'.stdout')+b'\n'+raw('controller/'+name+'.stderr');runs=re.findall(rb'^Ran (\d+) tests? in ([0-9.]+)s$',b,re.M);assert len(runs)==1 and int(runs[0][0])==count and len(re.findall(rb'^OK$',b,re.M))==1 and not re.search(rb'^(?:FAILED|FAIL:|ERROR:|OK \()',b,re.M)
 for name in ['nanoc_stage1','nanoc_stage2']:
  r=audit['artifacts'][name];b=raw('products/bin/'+name);assert b[:4]==bytes.fromhex('cffaedfe') and int.from_bytes(b[4:8],'little')==0x0100000c and int.from_bytes(b[12:16],'little')==2
  assert r['magic']=='cffaedfe' and r['cputype']==0x0100000c and r['filetype']==2 and r['sha256']==receipt['artifacts'][name]
 assert b'bin/nanoc succeeded with bin/nanoc_c removed' in raw('controller/bootstrap.stdout')
 assert '.bootstrap3.built' in audit['sentinels']
 accepted=True
print(json.dumps(dict(verified=True,records=len(records),source_blobs_verified=source_checks,source_pin=q['source_pin'],darwin_factory_accepted=accepted,phases=q['phases'],whole_release=False,pure_translator=False,paired_translator=False,current_process_liveness_checked=False)))
