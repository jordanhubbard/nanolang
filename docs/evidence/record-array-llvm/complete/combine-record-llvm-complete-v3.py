from pathlib import Path
import hashlib,json,os,shutil
seals=[Path('/tmp')/('nanolang-record-'+n+'-seal') for n in ['llvm-linux-ordinary','llvm-linux-wasm','llvm-darwin-recovery','package-425bc-final','llvm-linux-sanitizer']]
seals[-1]=Path('/run/user/1000/nanolang-record-llvm-linux-sanitizer-seal')
for source in seals:
 assert (source/'summary.json').is_file(),str(source)
final=json.loads(Path('/tmp/nanolang-record-llvm-linux-sanitizer-finalization-v3.json').read_text());assert final['state']=='sealed'
out=Path('/run/user/1000/nanolang-record-llvm-complete-seal');out.mkdir(exist_ok=False);(out/'reports').mkdir();(out/'objects').mkdir()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
reports={};objects={};summaries=[]
for source in seals:
 index=json.loads((source/'report-sha256.json').read_text());store=json.loads((source/'artifact-store.json').read_text());summary=json.loads((source/'summary.json').read_text());summaries.append({'seal':source.name,**summary})
 for name,row in index.items():
  p=source/'reports'/name;assert digest(p)==row['sha256'] and p.stat().st_size==row['bytes'];dst=out/'reports'/source.name/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);reports[str(dst.relative_to(out/'reports'))]=row
 for name in ['report-sha256.json','artifact-store.json','summary.json']:
  p=source/name;dst=out/'reports'/source.name/name;shutil.copyfile(p,dst);reports[str(dst.relative_to(out/'reports'))]={'sha256':digest(dst),'bytes':dst.stat().st_size}
 for h,row in store.items():
  p=source/'objects'/h;assert digest(p)==h and p.stat().st_size==row['bytes'];dst=out/'objects'/h
  if not dst.exists():
   try:os.link(p,dst)
   except OSError:shutil.copyfile(p,dst)
  assert digest(dst)==h
  objects[h]={'bytes':p.stat().st_size,'path':str(dst)}
summary={'source':'7804258431ed8fba21fa33c5507b06ad9a1bbd94','package_correction':'425bcbc786d00f171adb66cdc58af0783f23ef1b','native_optimization_fixture':'aed76cdb3','reports':len(reports),'objects':len(objects),'object_bytes':sum(r['bytes'] for r in objects.values()),'references':sum(s['references'] for s in summaries),'equal_pairs':sum(s['equal_pairs'] for s in summaries),'histories':summaries,'scope':'Exact complete retained selected histories; earlier remote-only evidence lost at reboot remains explicitly outside passing retained gates. Public/source/full graphs remain required.'}
for name,value in [('report-sha256.json',reports),('artifact-store.json',objects),('summary.json',summary)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='histories'},indent=2))
