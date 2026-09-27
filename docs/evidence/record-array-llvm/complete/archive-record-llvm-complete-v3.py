from pathlib import Path
import hashlib,json,os,shutil,stat,subprocess,sys,tarfile
root=Path('/run/user/1000/nanolang-record-llvm-complete-seal');archive=root.parent/'nanolang-record-llvm-complete-evidence.tar.gz';manifest=root.parent/'nanolang-record-llvm-complete-archive-manifest.json'
assert not archive.exists() and shutil.disk_usage(root).free>4*1024**3
entries={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for p in root.rglob('*'):
 if p.is_dir():continue
 assert p.is_file() and not p.is_symlink()
 entries[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)}
manifest.write_text(json.dumps(entries,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as tar:
 for name in sorted(entries):tar.add(root/name,arcname=name,recursive=False)
verifier=Path('/tmp/verify-record-llvm-o2-archive.py')
local=json.loads(subprocess.check_output([sys.executable,str(verifier),str(archive),str(manifest)],text=True,timeout=1800))
remote='/Users/jkh/nanolang-qualification/vm-effects-llvm-complete-20260921'
subprocess.run(['ssh','-o','ConnectTimeout=15','puck.local','mkdir -p '+remote],check=True,timeout=60)
for p in (archive,manifest,verifier):subprocess.run(['scp','-q',str(p),'puck.local:'+remote+'/'+p.name],check=True,timeout=1800)
actual=json.loads(subprocess.check_output(['ssh','-o','ConnectTimeout=15','puck.local','python3 '+remote+'/'+verifier.name+' '+remote+'/'+archive.name+' '+remote+'/'+manifest.name],text=True,timeout=1800));assert actual==local
result={'archive':str(archive),'bytes':archive.stat().st_size,'local':local,'remote':actual,'durable_remote':remote,'members':len(entries),'temporary_removed':False}
p=Path('/tmp/nanolang-record-llvm-complete-archive-verification.json');p.write_text(json.dumps(result,indent=2)+'\n')
subprocess.run(['scp','-q',str(p),'puck.local:'+remote+'/'+p.name],check=True,timeout=120)
print(json.dumps(result),flush=True)
