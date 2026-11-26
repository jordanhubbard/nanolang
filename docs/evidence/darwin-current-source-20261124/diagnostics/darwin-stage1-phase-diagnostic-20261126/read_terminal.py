"""I replay archived phase evidence and keep all compiler/release acceptance false."""
from pathlib import Path
import argparse,json,gzip,hashlib,subprocess,sys
parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path);parser.add_argument('--parent',type=Path);args=parser.parse_args();base=Path(__file__).resolve().parent;parent=args.parent or base.parent/'darwin-stage1-generation-timeout-20261126';sha=lambda b:hashlib.sha256(b).hexdigest();m=json.loads((base/'manifest.json').read_bytes());rows={r['path']:r for r in m['files']};assert len(rows)==len(m['files'])
def raw(name):
 r=rows[name];p=Path(r['stored']);assert not p.is_absolute() and '..' not in p.parts;packed=(base/p).read_bytes();assert len(packed)==r['stored_bytes'] and sha(packed)==r['stored_sha256'];b=gzip.decompress(packed);assert len(b)==r['bytes'] and sha(b)==r['sha256'],name;return b
for name in rows:raw(name)
load=lambda n:json.loads(raw(n));q=load('controller/result.json');request=load('controller/request.json');audit=load('terminal-audit.json');frozen=load('frozen-parent-audit.json')
assert q['terminal'] and not q['accepted'] and not q['whole_release'] and q['diagnostic_only'] and not m['accepted'] and not m['whole_release']
assert q['source_pin']==request['source_pin']==m['source_pin']=='d907f264d16d348b4589d3760bd213803e1c6f35'
assert request['parent_publication']==m['parent_publication']=='01332e4c1ccf7beffdea50aea63690cbff036ad5'
assert sha((parent/'manifest.json').read_bytes())==request['parent_manifest_sha256']==m['parent_manifest_sha256']
assert sha((parent/'supplement.json').read_bytes())==request['parent_supplement_sha256']==m['parent_supplement_sha256']
assert sha((parent/'read_terminal.py').read_bytes())==request['parent_reader_sha256']
command=[sys.executable,str(parent/'read_terminal.py')]+(['--repo',str(args.repo)] if args.repo else []);check=subprocess.run(command,capture_output=True,text=True);assert check.returncode==0,check.stderr;parent_result=json.loads(check.stdout);assert parent_result['verified'] and not parent_result['darwin_factory_accepted']
pm=json.loads((parent/'manifest.json').read_bytes());ps=json.loads((parent/'supplement.json').read_bytes());pr={r['path']:r for r in pm['files']+ps['files']}
def parentraw(name):return gzip.decompress((parent/pr[name]['stored']).read_bytes())
pq=json.loads(parentraw('controller/result.json'))
assert all(sha(raw('controller/'+n))==h for n,h in request['observer_sources'].items())
assert (request['seed_seconds'],request['verify_seconds'],request['observation_seconds'],request['compiler_shadow_seconds'],request['ordinary_seconds'],request['original_qualification_stage_seconds'])==(600,60,300,30,10,1800)
for command in q['commands']:
 for stream in ['stdout','stderr']:
  if stream+'_sha256' in command:assert sha(raw('controller/'+command['label']+'.'+stream))==command[stream+'_sha256']
for a in q['admissions']:
 assert a['passed'] and a['filesystem']=='apfs' and a['free_bytes']>15*1024**3 and not a['other_nanolang_jobs'] and len(a['cpu'])==2 and all(c['share']>=.75 for c in a['cpu'])
assert q['source_before']==q['source_after']==q['parent_sources_before']==q['parent_source_after']==pq['source_after']
assert q['images_before']==q['images_after']==q['parent_images_before']==q['parent_images_after']==pq['images_after']
assert q['tools_before']==q['tools_after']==pq['tools_after']
assert q['parent_result_before']==q['parent_result_after']==sha(parentraw('controller/result.json'))
assert q['guards_match'] and not q['guard_errors'] and not q['remaining_owned_processes'] and q['reservation_released']
assert frozen['match'] and frozen['expected']==frozen['actual']
assert audit['source_match'] and audit['images_match'] and audit['tools_match'] and audit['new_hosts_match'] and not audit['remaining_owned_processes'] and not audit['reservation_exists']
probe=audit['controller_probe'];assert (probe['exit']==1 and not probe['stdout']) or (probe['exit']==0 and probe['stdout'].strip() and not probe['stdout'].strip().startswith(q['process_start']))
namespace={};exec(raw('controller/seed_comparison.py').decode(),namespace)
prefix=load('controller/prefix-result.json');r=prefix['manifest'];assert prefix['prepared_prefix_only'] and prefix['stopped_before_stage1'] and not prefix['bootstrap_accepted'] and not prefix['whole_release']
assert prefix['budgets']==[{'label':'seed','seconds':600},{'label':'seed-verify','seconds':60}]
assert r['phase']=='building-stage1' and r['stage_timeout_seconds']==600 and r['shadow_timeout_seconds']==30
assert [x['label'] for x in r['steps']]==['seed','seed-verify'] and all(x['exit_code']==0 for x in r['steps'])
for step in r['steps']:assert sha(raw('bootstrap/'+Path(r['work']).name+'/'+Path(step['log']).name))==step['log_sha256']
assert set(r['artifacts'])=={'nanoc_seed.nvm'} and q['new_seed_before']==q['new_seed_after']==r['artifacts']['nanoc_seed.nvm']==sha(raw('products/bin/nanoc_seed.nvm'))
assert raw('products/bin/nanoc_seed.nvm').startswith(b'NVM\x02')
comparison=namespace['compare'](parentraw('products/bin/nanoc_seed.nvm'),raw('products/bin/nanoc_seed.nvm'),json.loads(parentraw('bootstrap/working-receipt.json')),r);assert comparison==load('seed-comparison.json') and comparison['only_eight_bound_path_changes'] and comparison['all_other_sections_equal'] and not comparison['whole_modules_equal']
assert q['new_hosts_before']==q['new_hosts_after']==r['hosts'] and len(r['hosts'])==6
for name,h in r['hosts'].items():assert sha(raw('hosts/'+str(Path(name).relative_to(Path(q['root'])/'source'))))==h
assert q['working_sources_before']==q['working_sources_after']==r['sources']
extra=set(r['sources'])-set(q['source_after']);assert extra=={'modules/index.json'}
for n,h in r['sources'].items():assert (sha(raw('inputs/'+n)) if n in extra else q['source_after'][n])==h
assert q['phases']==['seed-prefix','bounded-observation'] and not q.get('exception') and not q['unexpected_native_compiler'] and q['diagnostic_valid']
assert [x['label'] for x in q['commands']]==['parent-reader','seed-prefix','phase-observation']
assert all(x['exit']==0 for x in q['commands'][:2]);last=q['commands'][-1];assert last['timeout']==300
if last.get('timed_out'):assert last['seconds']>=300 and last['exit']!=0
assert q['observation_environment']['NANOLANG_BOOTSTRAP_NO_CC']=='1' and q['observation_environment']['NANO_SHADOW_TIMEOUT_SECONDS']=='30'
events=[json.loads(line) for line in raw('controller/phases.jsonl').splitlines()];assert events==q['phase_events'] and events
assert all(isinstance(e['phase'],str) and e['phase'] and e['monotonic']>=q['observation_started_monotonic'] for e in events)
assert all(a['monotonic']<=b['monotonic'] for a,b in zip(events,events[1:]))
print(json.dumps(dict(verified=True,records=len(rows),parent_records=parent_result['records'],source_blobs_verified=parent_result['source_blobs_verified'],diagnostic_valid=True,phase_events=[dict(phase=e['phase'],seconds=e['monotonic']-q['observation_started_monotonic']) for e in events],observation_exit=last['exit'],observation_timed_out=last.get('timed_out',False),compiler_accepted=False,whole_release=False,current_process_liveness_checked=False)))
