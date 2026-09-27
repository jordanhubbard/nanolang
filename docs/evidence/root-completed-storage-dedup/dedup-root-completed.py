from pathlib import Path
import os,stat,hashlib,json
base=Path('/home/jkh/nanolang-qualification')
roots=[base/'frame-bindings-20260921',base/'artifact-native-link-flags']+list(base.glob('capture-transport-*-linux*'))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
groups={};inodes=set();plan=[]
for root in roots:
 for p in root.rglob('*'):
  if '.git' in p.parts or p.is_symlink():continue
  s=p.stat()
  if not stat.S_ISREG(s.st_mode) or s.st_size<32768 or (s.st_dev,s.st_ino) in inodes:continue
  inodes.add((s.st_dev,s.st_ino))
  key=(s.st_dev,s.st_size,stat.S_IMODE(s.st_mode),s.st_uid,s.st_gid,digest(p))
  if key not in groups:groups[key]=p;continue
  source=groups[key]
  if os.listxattr(p)!=os.listxattr(source) or any(os.getxattr(p,x)!=os.getxattr(source,x) for x in os.listxattr(p)):continue
  plan.append({'source':str(source),'destination':str(p),'sha256':key[-1],'bytes':s.st_size,'mode':key[2],'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'links':s.st_nlink,'blocks':s.st_blocks})
(base/'dedup-root-completed-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'candidates':len(plan),'potential_bytes':sum(r['blocks']*512 for r in plan if r['links']==1)}),flush=True)
reclaimed=0
for r in plan:
 p=Path(r['destination']);source=Path(r['source']);s=p.stat()
 assert s.st_ino==r['inode'] and digest(p)==r['sha256'] and digest(source)==r['sha256']
 tmp=p.with_name(p.name+'.dedup-link')
 assert not tmp.exists()
 os.link(source,tmp);os.replace(tmp,p)
 assert digest(p)==r['sha256'] and stat.S_IMODE(p.stat().st_mode)==r['mode']
 if s.st_nlink==1:reclaimed+=s.st_blocks*512
result={'files':len(plan),'reclaimed_bytes':reclaimed,'all_paths_bytes_modes_preserved':True,'scope':'Completed root-owned qualification artifacts only; original destination mtimes retained in plan.'}
(base/'dedup-root-completed-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
