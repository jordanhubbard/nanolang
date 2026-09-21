from pathlib import Path
import hashlib,json,subprocess,sys,tarfile,stat,time,shutil
base=Path(sys.argv[1]).resolve();assert (base/'outer.json').is_file();assert json.loads((base/'outer.json').read_text())['returncode']==1;assert json.loads((base/'package-hb-outer.json').read_text())['returncode']==0
assert shutil.disk_usage(base).free>=2147483648
python=sys.executable
subprocess.run([python,str(base/'audit-backend-integration-coverage-puck.py'),str(base)],check=True,timeout=300)
subprocess.run([python,str(base/'verify-record-backend-current.py'),'--root',str(base/'source'),'--report',str(base/'package-hb-reports'),'--output',str(base/'current-inputs.json')],check=True,timeout=600)
seal=base/'seal';args=[python,str(base/'seal-record-llvm-local-copies.py'),'--output',str(seal),'--history',str(base/'reports'),'--history',str(base/'package-hb-reports')]
for name in ('source.json','launch.json','outer.json','outer.log','current-inputs.json','coverage-audit.json','nanolang-backend-integration-launch.py','nanolang-record-backend-integration-driver.py','nanolang-record-package-retain.py','audit-backend-integration-coverage-puck.py','verify-record-backend-current.py','finalize-backend-integration-puck.py','seal-record-llvm-local-copies.py','first-extraction-terminal.json','staged-selector-preflight.json','orchestration-inputs.json','integration-correspondence.json','nanolang-backend-package-hb-launch.py','package-hb-launch.json','package-hb-outer.json','package-hb-outer.log','package-hb-original-current.json'):
 p=base/name
 if p.is_file():args+=['--extra',str(p)]
subprocess.run(args,check=True,timeout=1800)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
index={}
for p in seal.rglob('*'):
 if p.is_file():index[str(p.relative_to(seal))]={'sha256':sha(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)}
archive=base/'sealed-evidence.tar.gz';manifest=base/'sealed-evidence-manifest.json';manifest.write_text(json.dumps(index,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as tar:
 for n in sorted(index):tar.add(seal/n,arcname=n,recursive=False)
with tarfile.open(archive) as tar:
 assert set(tar.getnames())==set(index)
 for item in tar:
  row=index[item.name];assert item.isfile() and item.size==row['bytes'] and item.mode==row['mode'];h=hashlib.sha256()
  with tar.extractfile(item) as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  assert h.hexdigest()==row['sha256']
record={'status':'PASS','archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'members':len(index),'manifest_sha256':sha(manifest),'source':json.loads((base/'source.json').read_text())['pin'],'finished':time.time(),'scope':'Local archive verification; independent durable peer copy remains required before temporary-data cleanup.'}
(base/'archive-verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
