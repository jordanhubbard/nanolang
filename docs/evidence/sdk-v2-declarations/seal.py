from pathlib import Path
import hashlib,json,tarfile,shutil
b=Path('/home/jkh/nanolang-qualification');d=b/'declaration-v2-8bf-drivers';dest=Path('/home/jkh/Src/nanolang-5.1-candidate/docs/evidence/sdk-v2-declarations');dest.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
copy=d/'seal-puck-copy';copy.mkdir(exist_ok=True)
with tarfile.open(d/'puck-reports.tar.gz') as tf:tf.extractall(copy,filter='data')
rows={}
for host,root in [('linux',b/'declaration-v2-8bf-linux'),('puck',copy)]:
 ident=load(root/'identity.json');phases=load(root/'phases.json');assert load(root/'terminal.json')['status']==0
 assert ident['driver_sha']==sha(d/'run.py') and ident['archive_sha']==sha(d/'source.tar.gz')
 assert len(phases)==4 and all(p['returncode']==0 and not p['timeout'] and not p['remaining_groups'] for p in phases)
 assert load(root/'tools-after.json')==ident['tools']
 for mode in ('ordinary','sanitized'):
  p=root/mode;assert load(p/'source-before.json')==load(p/'source-after.json')
  assert load(p/'providers-before.json')==load(p/'providers-after.json')
  log=(p/(mode+'-declarations.log')).read_text();assert 'PASS 1987 complete mixed declaration checks' in log and 'PASS 14279 complete mixed declaration checks' in log
  statuses=list((p/'tmp').rglob('*-status.json'));assert len(statuses)==4
  for status in statuses:
   x=load(status);assert x['returncode']==0 and not x['timeout'] and x['leader_reaped'] and x['group_disappeared']
 archive=dest/(host+'.tar.gz');members={}
 with tarfile.open(archive,'w:gz') as tf:
  for p in root.rglob('*'):
   if p.is_file() and 'source' not in p.relative_to(root).parts and p.suffix in ('.json','.log','.txt','.mk'):
    name=str(p.relative_to(root));tf.add(p,arcname=name);members[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
 with tarfile.open(archive) as tf:assert {m.name:{'bytes':m.size,'sha256':hashlib.sha256(tf.extractfile(m).read()).hexdigest()} for m in tf.getmembers()}==members
 rows[host]={'pin':ident['pin'],'archive_sha256':sha(archive),'members':members,'phases':phases}
for name in ('run.py','scope.json'):shutil.copy2(d/name,dest/name)
(dest/'seal.json').write_text(json.dumps(rows,indent=2)+'\n')
print(sum(len(x['members']) for x in rows.values()))
