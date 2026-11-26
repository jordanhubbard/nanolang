"""I verify setup failures and a dated live observation, with no product admission."""
from pathlib import Path
import hashlib,json
base=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest();load=lambda name:json.loads((base/name).read_bytes())
for item in load('manifest.json')['files']:
 raw=(base/item['path']).read_bytes();assert len(raw)==item['bytes'] and sha(raw)==item['sha256'],item['path']
first=load('failed-filesystem/result.json');second=load('failed-linux-target/result.json');live=load('live-observation/result.json');process=load('live-observation/process-observation.json');lease=load('live-observation/reservation-owner.json')
assert first['terminal'] and not first['accepted'] and not first['commands'] and not first.get('reservation')
assert first['admissions'][0]['filesystem']=='/' and first['admissions'][0]['free_bytes']>15*1024**3 and not first['admissions'][0]['other_nanolang_jobs']
assert not first['remaining_owned_processes']
assert second['terminal'] and not second['accepted'] and second['guards_match'] and second['reservation_released'] and not second['remaining_owned_processes']
assert second['source_before']==second['source_after'] and len(second['source_before'])==4683
assert second['tools_before']==second['tools_after'] and len(second['tools_before'])==9
assert second['phases']==[] and second['commands'][-1]['label']=='cold-tools' and second['commands'][-1]['exit']==2
assert 'bin/nano_as_capture.so' in second['commands'][-1]['argv']
assert b'No rule to make target' in (base/'failed-linux-target/cold-tools.stderr').read_bytes()
assert b'bin/nano_as_capture.so' in (base/'failed-linux-target/cold-tools.stderr').read_bytes()
assert not live['terminal'] and not live['accepted'] and not live['phases']
assert process['exit']==0 and process['pid']==live['pid']==6518 and process['expected_start']==live['process_start']
assert process['observation'].startswith(live['process_start']) and live['root'] in process['observation']
assert lease['root']==live['root'] and lease['pid']==live['pid'] and lease['process_start']==live['process_start']
for state in [second,live]:
 admission=state['admissions'][0];assert admission['passed'] and admission['filesystem']=='apfs' and admission['free_bytes']>15*1024**3 and not admission['other_nanolang_jobs']
 assert len(admission['cpu'])==2 and all(r['share']>=.75 for r in admission['cpu'])
 assert admission['filesystem_probe']['device']=='/dev/disk3s1' and all('(apfs,' in row for row in admission['filesystem_probe']['mounts'])
for state in [first,second,live]:
 assert state['source_pin']=='d907f264d16d348b4589d3760bd213803e1c6f35' and not state['whole_release']
 assert (state['stage_seconds'],state['compiler_shadow_seconds'],state['ordinary_seconds'],state['emitter_helper_seconds'])==(1800,30,10,300)
for row in second['commands']:
 assert sha((base/'failed-linux-target'/(row['label']+'.stdout')).read_bytes())==row['stdout_sha256']
 assert sha((base/'failed-linux-target'/(row['label']+'.stderr')).read_bytes())==row['stderr_sha256']
products=load('failed-linux-target/partial-products.json');assert len(products)==7
assert all(not row['symlink'] and row['bytes']>0 and row['magic']=='cffaedfe' for row in products.values())
old=(base/'failed-filesystem/control.py').read_text();middle=(base/'failed-linux-target/control.py').read_text();current=(base/'live-observation/control.py').read_text()
needle=' record=dict(label=label,utc='
insert=""" df=subprocess.check_output(['/bin/df','-P',str(root)],text=True);device=df.splitlines()[1].split()[0]
 selected=[line for line in subprocess.check_output(['/sbin/mount'],text=True).splitlines() if line.startswith(device+' on ')]
 types={line.rsplit(' (',1)[1].split(',',1)[0].rstrip(')') for line in selected}
 filesystem='apfs' if types=={'apfs'} else ','.join(sorted(types))
"""
assert old.replace(needle,insert+needle,1).replace("filesystem=subprocess.check_output(['stat','-f','%T',str(root)],text=True).strip()","filesystem=filesystem,filesystem_probe=dict(device=device,mounts=selected)")==middle
assert middle.count(",'bin/nano_as_capture.so'")==2 and middle.replace(",'bin/nano_as_capture.so'",'')==current
assert all('bootstrap' not in row['label'] and row['label'] not in ['native36','emitter12'] for row in live['commands'])
print(json.dumps(dict(verified=True,setup_attempts_failed=2,products_qualified=False,release_complete=False,live_observation_utc=process['captured_utc'],recorded_controller_pid=6518,recorded_controller_start=process['expected_start'],current_liveness_checked=False)))
