import hashlib, importlib.util, json, sys, time, zipfile
from pathlib import Path
root=Path('/Users/jkh/Src/nanolang-51-harness')
sys.path.insert(0,str(root/'scripts'))
spec=importlib.util.spec_from_file_location('checkpoint',root/'scripts/ci_bootstrap_checkpoint.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
z=Path('/private/tmp/nanolang-coverage-checkpoint-12204047789.zip')
out=Path('/private/tmp/nanolang-coverage-checkpoint-12204047789.tar.gz')
start=time.time(); h=hashlib.sha256()
with z.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
assert h.hexdigest()=='98061e6c5df0aac49260d9d8e5f7148ac425df2aeb44d6fcfca824b57b8bb79e'
print('Official ZIP SHA256 verified',flush=True)
with zipfile.ZipFile(z) as a:
 assert a.namelist()==['coverage-bootstrap.tar.gz']
 with a.open(a.namelist()[0]) as src,out.open('wb') as dst:
  for b in iter(lambda:src.read(1024*1024),b''):dst.write(b)
print('ZIP member CRC verified; validating all tar member records',flush=True)
manifest=m.read_archive(out,Path('/home/runner/work/nanolang/nanolang'))
r={'artifact_id':12204047789,'zip_sha256':h.hexdigest(),'zip_bytes':z.stat().st_size,'tar_gz_bytes':out.stat().st_size,'commit':manifest['commit'],'phase':manifest['phase'],'records':len(manifest['files']),'directories':sum(v['kind']=='directory' for v in manifest['files'].values()),'elapsed_seconds':time.time()-start,'restored_or_executed':False}
Path('/private/tmp/nanolang-coverage-checkpoint-12204047789-validated.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r),flush=True)
