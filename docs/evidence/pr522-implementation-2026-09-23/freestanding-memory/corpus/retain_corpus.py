from pathlib import Path
import argparse,hashlib,json,tarfile,gzip,io
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');p.add_argument('label');a=p.parse_args()
root=Path(a.root);output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
rows=[];reports=[]
for f in sorted(root.rglob('*')):
 if not f.is_file():continue
 if f.is_symlink():raise ValueError('I require regular retained artifacts: '+str(f))
 data=f.read_bytes();name=str(f.relative_to(root));h=hashlib.sha256(data).hexdigest()
 rows.append({'path':name,'bytes':len(data),'sha256':h})
 if f.suffix in ('.json','.log','.txt'):reports.append((name,data,h))
(output/(a.label+'-inventory.json')).write_text(json.dumps({'root':str(root),'files':rows},indent=2)+'\n')
archive=output/(a.label+'-reports.tar.gz')
with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
 for name,data,h in reports:
  info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(data))
expected={name:h for name,data,h in reports}
with tarfile.open(archive,'r:gz') as tar:
 actual={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}
assert actual==expected
summary={'files':len(rows),'reports':len(reports),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'reports_rehashed':True}
(output/(a.label+'-archive.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
