import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/home/jkh/Src/nanolang-mixed-generated-contract');base=root/'docs/evidence/record-array-execution-plan';base.mkdir(parents=True,exist_ok=False)
objects={};summaries={}
for host,directory in [('linux','/tmp/nanolang-record-execution-linux-seal'),('puck','/tmp/nanolang-record-execution-puck-seal-local')]:
 p=Path(directory);shutil.copytree(p/'reports',base/host)
 for name in ['summary.json','artifact-store.json','report-sha256.json']:shutil.copyfile(p/name,base/host/name)
 summaries[host]=json.loads((p/'summary.json').read_text())
 for h,v in json.loads((p/'artifact-store.json').read_text()).items():
  q=p/'objects'/h;assert hashlib.sha256(q.read_bytes()).hexdigest()==h
  e=objects.setdefault(h,{'bytes':v['bytes'],'copies':[]});assert e['bytes']==v['bytes'];e['copies'].append(str(q))
pin='c032c0a21730b6abca4ce71add2048c96cbb36d9'
selected=subprocess.check_output(['git','diff','--name-only','ad0080eb',pin],cwd=root,text=True).splitlines()
selected=[p for p in selected if not p.startswith('docs/')]
identity={}
for name in selected:
 data=subprocess.check_output(['git','show',pin+':'+name],cwd=root);assert (root/name).read_bytes()==data
 identity[name]={'sha256':hashlib.sha256(data).hexdigest(),'git_blob':subprocess.check_output(['git','rev-parse',pin+':'+name],cwd=root,text=True).strip()}
archive=Path('/tmp/nanolang-record-execution-puck-evidence.tar.gz')
summary={'source_pin':pin,'production_correction':'429da44bcf0f3803b4964d0163660d50565faa34','objects':len(objects),'object_bytes':sum(v['bytes'] for v in objects.values()),'references':sum(v['references'] for v in summaries.values()),'equal_source_tool_pairs':sum(v['equal_pairs'] for v in summaries.values()),'phase_terminals':sum(len(v['terminals']) for v in summaries.values()),'preserved_nonzero_terminals':[dict(host=host,**t) for host,v in summaries.items() for t in v['terminals'] if t['returncode']!=0],'darwin_archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},'selected_source':identity,'scope':'copied non-admitting execution-plan preparation only; seven query compiler configurations, linked providers partly ordinary; no generated consumer/runtime/public/source admission'}
for name,v in [('objects.json',objects),('summary.json',summary)]: (base/name).write_text(json.dumps(v,indent=2)+'\n')
for script in ['/tmp/seal-record-execution-plan.py',__file__]:shutil.copyfile(script,base/Path(script).name)
manifest={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(base.rglob('*')) if p.is_file()}
(base/'seal-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'reports':len(manifest),**{k:v for k,v in summary.items() if k not in ['selected_source','preserved_nonzero_terminals']}},indent=2))
