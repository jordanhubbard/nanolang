from pathlib import Path
import hashlib,json,subprocess,tarfile,shutil,stat
base=Path('/run/user/1000/nanolang-capture-file-d1f-linux-r2')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
checks=json.loads((base/'reports/checks.json').read_text());assert checks['terminal']['returncode']==0 and checks['source_unchanged'] and checks['tools_unchanged']
selected={n:base/n for n in ['run.py','paths.json','reference.pack','source.tar.gz']}
for p in (base/'reports').rglob('*'):
 if p.is_file():selected[str(p.relative_to(base))]=p
products=json.loads((base/'reports/products.json').read_text())
for n,row in products.items():
 p=base/n if n.startswith('artifacts/') else base/'source'/n
 assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],n
 selected[str(p.relative_to(base))]=p
# I retain the first failed setup separately, without claiming it passed.
first=base.parent/'nanolang-capture-file-d1f-linux'
for p in (first/'reports').rglob('*'):
 if p.is_file():selected['first-setup/'+str(p.relative_to(first))]=p
selected['first-setup/run.py']=first/'run.py'
entries={}
for n,p in selected.items():
 assert not p.is_symlink(),n
 entries[n]={'sha256':sha(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)}
archive=base.parent/'capture-file-d1f-linux-evidence.tar.gz';index=base.parent/'capture-file-d1f-linux-manifest.json';index.write_text(json.dumps(entries,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as t:
 for n in sorted(selected):t.add(selected[n],arcname=n,recursive=False)
verifier=Path('/tmp/verify-record-llvm-o2-archive.py')
local=json.loads(subprocess.check_output(['python3',str(verifier),str(archive),str(index)],text=True))
remote='/Users/jkh/nanolang-qualification/capture-file-d1f-linux-retained'
subprocess.run(['ssh','-o','BatchMode=yes','jkh@puck.local','mkdir',remote],check=True)
for p in [archive,index,verifier]:subprocess.run(['scp','-q',str(p),'jkh@puck.local:'+remote+'/'],check=True)
other=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','jkh@puck.local','/opt/homebrew/bin/python3.14',remote+'/'+verifier.name,remote+'/'+archive.name,remote+'/'+index.name],text=True));assert local==other
result={'local':local,'remote':other,'durable_root':remote,'checks':checks,'source':'d1f035314','temporary_data_removed':False}
report=Path('/home/jkh/nanolang-qualification/capture-file-d1f-linux-retention.json');report.write_text(json.dumps(result,indent=2)+'\n')
subprocess.run(['scp','-q',str(report),'jkh@puck.local:'+remote+'/retention.json'],check=True)
print(json.dumps(result),flush=True)
