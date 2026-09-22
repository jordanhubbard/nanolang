from pathlib import Path
import json,hashlib,tarfile,gzip,io,os,subprocess
base=Path('/home/jkh/nanolang-qualification');out=base/'backend-current-941-seal';out.mkdir(exist_ok=False);(out/'objects').mkdir();(out/'reports').mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
index={'pin':'2a41f3f56635d84d64aa57f3ca556f5ad33aa102','reports':[],'objects':{},'refs':[],'pairs':[],'histories':[]}
for host in ['linux','puck']:
 root=base/('backend-current-941-linux' if host=='linux' else 'backend-current-941-puck-local');report=root/'reports';rows=[]
 for p in (report/'artifacts').iterdir():
  data=p.read_bytes();h=sha(data);assert h==p.name
  dest=out/'objects'/h
  if not dest.exists():os.link(p,dest)
  index['objects'][h]={'bytes':len(data)}
 for p in report.glob('*-artifacts.json'):
  for name,v in json.loads(p.read_text()).items():
   assert v['sha256'] in index['objects'];index['refs'].append({'host':host,'map':p.name,'path':name,'sha256':v['sha256']})
 for suffix in ['source','tools']:
  for p in report.glob('*-'+suffix+'-before.json'):
   after=p.with_name(p.name.replace('-before.json','-after.json'));assert json.loads(p.read_text())==json.loads(after.read_text());index['pairs'].append({'host':host,'before':p.name,'after':after.name,'count':len(json.loads(p.read_text()))})
 for p in report.glob('*-terminal.json'):
  d=json.loads(p.read_text());assert d['returncode']==0 and not d['timeout'] and d['leader_reaped'] and d['group_disappeared'] and not d['errors']
 for p in report.glob('*-nested-processes.json'):assert not json.loads(p.read_text())['remaining']
 assert len(json.loads((report/'results.json').read_text()))==4
 assert json.loads((report/'c-baseline-before.json').read_text())==json.loads((report/'c-baseline-after.json').read_text())
 bundle=out/'reports'/(host+'.tar.gz')
 selected=list(report.rglob('*'))+[root/n for n in ['source.json','outer.json','outer.log','launch.json','staging.json','first-staging-terminal.json','nanolang-backend-current-launch.py','nanolang-record-backend-current-driver.py','nanolang-record-package-retain.py']]
 with bundle.open('wb') as f,gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
  for p in sorted(set(selected)):
   if not p.is_file() or 'artifacts' in p.relative_to(root).parts:continue
   data=p.read_bytes();name=p.relative_to(root).as_posix();entry={'host':host,'path':name,'bytes':len(data),'sha256':sha(data)};rows.append(entry);info=tarfile.TarInfo(name);info.size=len(data);info.mode=p.stat().st_mode&0o777;tar.addfile(info,io.BytesIO(data))
 with tarfile.open(bundle) as tar:
  for e in rows:assert sha(tar.extractfile(e['path']).read())==e['sha256']
 index['reports']+=rows;index['histories'].append({'host':host,'bundle':bundle.name,'sha256':sha(bundle.read_bytes()),'bytes':bundle.stat().st_size})
index['object_bytes']=sum(v['bytes'] for v in index['objects'].values());index['puck_archive']={'path':str(base/'backend-current-941-puck-evidence.tar.gz'),'sha256':sha((base/'backend-current-941-puck-evidence.tar.gz').read_bytes())}
(out/'reports'/'seal.json').write_text(json.dumps(index,indent=2)+'\n');print({k:len(index[k]) for k in ['reports','objects','refs','pairs']},index['object_bytes'])
