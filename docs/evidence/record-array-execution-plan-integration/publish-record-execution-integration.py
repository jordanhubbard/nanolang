import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-record-array-execution-ready');base=root/'docs/evidence/record-array-execution-plan-integration';base.mkdir(parents=True,exist_ok=False)
objects={};summaries={}
for host,directory in [('linux','/tmp/nanolang-record-execution-integration-linux-seal'),('puck','/tmp/nanolang-record-execution-integration-puck-seal-local')]:
 p=Path(directory);shutil.copytree(p/'reports',base/host)
 for name in ['summary.json','artifact-store.json','report-sha256.json']:shutil.copyfile(p/name,base/host/name)
 summaries[host]=json.loads((p/'summary.json').read_text())
 for h,v in json.loads((p/'artifact-store.json').read_text()).items():
  q=p/'objects'/h;assert hashlib.sha256(q.read_bytes()).hexdigest()==h
  e=objects.setdefault(h,{'bytes':v['bytes'],'copies':[]});assert e['bytes']==v['bytes'];e['copies'].append(str(q))
pin='b87bb91f1793b48617d7737bcd4f6110590bbce1'
selected=subprocess.check_output(['git','diff','--name-only','128d5cd2',pin],cwd=root,text=True).splitlines()
selected=[p for p in selected if not p.startswith('docs/')]
identity={}
for name in selected:
 data=subprocess.check_output(['git','show',pin+':'+name],cwd=root);assert (root/name).read_bytes()==data
 identity[name]={'sha256':hashlib.sha256(data).hexdigest(),'git_blob':subprocess.check_output(['git','rev-parse',pin+':'+name],cwd=root,text=True).strip()}
archive=Path('/tmp/nanolang-record-execution-integration-puck-evidence.tar.gz')
summary={'source_pin':pin,'production_checkpoint':'f093b0ccc88f6e0b69861a8a986764bbfd50877e','objects':len(objects),'object_bytes':sum(v['bytes'] for v in objects.values()),'references':sum(v['references'] for v in summaries.values()),'equal_source_tool_pairs':sum(v['equal_pairs'] for v in summaries.values()),'phase_terminals':sum(len(v['terminals']) for v in summaries.values()),'preserved_nonzero_terminals':[dict(host=host,**t) for host,v in summaries.items() for t in v['terminals'] if t['returncode']!=0],'darwin_archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},'selected_source':identity,'scope':'copied non-admitting execution-plan preparation only; ordinary integrated query and named neighbors both hosts; original seven configurations remain separately attributed; no generated consumer/runtime/public/source admission'}
for name,v in [('objects.json',objects),('summary.json',summary)]: (base/name).write_text(json.dumps(v,indent=2)+'\n')
for script in ['/tmp/seal-record-execution-integration.py',__file__]:shutil.copyfile(script,base/Path(script).name)
manifest={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(base.rglob('*')) if p.is_file()}
(base/'seal-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'reports':len(manifest),**{k:v for k,v in summary.items() if k not in ['selected_source','preserved_nonzero_terminals']}},indent=2))
