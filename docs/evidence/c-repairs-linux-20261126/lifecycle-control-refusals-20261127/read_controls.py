"""I check selected control refusals; I do not reconstruct their whole ZIPs."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(root/'selection.json').read_bytes();assert sha(raw)=='c6deb2ce860a99e20bdeaa5cb9fddb5247959e2fcb9c0851cc160f3ee119503e'
selected=json.loads(raw);assert not selected['whole_release'];results=[]
for run in selected['runs']:
 p=root/str(run['run']);api=json.loads((p/'api-run.json').read_text());artifact=json.loads((p/'api-artifact.json').read_text())
 assert api['id']==run['run'] and api['head_sha']==run['head'] and api['conclusion']=='failure'
 assert artifact['id']==run['artifact'] and artifact['digest']=='sha256:'+run['downloaded_zip_sha256']
 assert run['nested_original_zip_omitted'] and not run['compiler_observation_ran']
 for entry in run['selected_entries']:
  name=Path(entry['path']);assert not name.is_absolute() and '..' not in name.parts
  data=(p/name).read_bytes();assert sha(data)==entry['sha256'] and len(data)==entry['bytes']
 sources=json.loads((p/'observer-sources.json').read_text())
 for path,h in sources.items():
  data=(p/'observer-sources'/Path(path).name).read_bytes();assert sha(data)==h
  assert data==subprocess.check_output(['git','show',run['head']+':'+path],cwd=root)
 build=json.loads((p/'observer-build.json').read_text());assert sha((p/'lifecycle-observer').read_bytes())==build['sha256']
 assert sha((p/'observer-sources/ci_darwin_shadow_lifecycle.c').read_bytes())==build['source_sha256']
 tree=json.loads((p/'product-tree-guard.json').read_text());assert tree['matches'] and not tree['qualification']
 assert not (p/'compiler.stderr').exists() and not (p/'controls.json').exists()
 controls={}
 for path in p.glob('control-*.jsonl'):
  rows=[json.loads(line) for line in path.read_text().splitlines()];last=rows[-1];assert last['event']=='control'
  kind=last['kind'];events=[v for v in rows if v['event']=='process-event']
  if last['passed']:
   assert last['wait_status']==0 and len(events)==1
   controls[kind]={'passed_c_outcome':True,'marker_after_origin_ns':last['marker_ns']-last['origin_ns'],'event_after_marker_ns':events[0]['after_ns']-last['marker_ns']}
  else:
   assert kind=='blocked' and last['marker_ns']==0 and last['wait_status']==9 and not events
   assert rows[-2]['event']=='end' and rows[-2]['result']==3
   controls[kind]={'passed_c_outcome':False,'marker_missing':True,'killed_after_ns':rows[-2]['monotonic_ns']-last['origin_ns']}
 results.append({'run':run['run'],'controls':controls,'compiler_observation':False})
assert results[0]['controls']['success']['marker_after_origin_ns']==178168000
assert results[0]['controls']['success']['event_after_marker_ns']==192000
assert results[1]['controls']['blocked']['marker_missing']
print(json.dumps({'verified_selected_bytes_and_sources':True,'runs':results,'whole_wrapper_zip_reconstructed':False,'qualification':False,'whole_release':False}))
