from pathlib import Path
import subprocess,shlex,hashlib,json,os,time
source=Path('/tmp/nanolang-file-service-parser-70113.tar');remote='/tmp/nanolang-file-service-parser-70113.retained.tar.gz'
script='''from pathlib import Path
import gzip,hashlib,json,os,shutil,sys
p=Path('/tmp/nanolang-file-service-parser-70113.retained.tar.gz');tmp=p.with_name(p.name+'.partial')
assert not p.exists() and not tmp.exists()
h=hashlib.sha256();length=0
with tmp.open('xb') as raw:
 with gzip.GzipFile(filename='nanolang-file-service-parser-70113.tar',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as out:
  while True:
   b=sys.stdin.buffer.read(4*1024*1024)
   if not b:break
   assert shutil.disk_usage(p.parent).free>2*1024**3
   h.update(b);length+=len(b);out.write(b)
 raw.flush();os.fsync(raw.fileno())
v=hashlib.sha256();verified=0
with gzip.open(tmp,'rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):v.update(b);verified+=len(b)
assert length==5881559040 and verified==length and v.digest()==h.digest()
c=hashlib.sha256()
with tmp.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):c.update(b)
row={'remote_path':str(p),'original_bytes':length,'original_sha256':h.hexdigest(),'compressed_bytes':tmp.stat().st_size,'compressed_sha256':c.hexdigest(),'decompressed_verified':True}
tmp.rename(p)
with p.with_name(p.name+'.json').open('w') as out:out.write(json.dumps(row,indent=2)+'\\n');out.flush();os.fsync(out.fileno())
print(json.dumps(row),flush=True)
'''
initial=source.stat();start=time.monotonic()
with source.open('rb') as inp:
 result=subprocess.run(['ssh','jkh@puck.local',shlex.join(['/usr/bin/python3','-c',script])],stdin=inp,capture_output=True,text=True,timeout=900)
Path('/tmp/nanolang-archive-remote.stdout').write_text(result.stdout);Path('/tmp/nanolang-archive-remote.stderr').write_text(result.stderr)
assert result.returncode==0,(result.returncode,result.stderr)
row=json.loads(result.stdout);h=hashlib.sha256()
with source.open('rb') as inp:
 for b in iter(lambda:inp.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==row['original_sha256'] and source.stat().st_mtime_ns==initial.st_mtime_ns and source.stat().st_size==row['original_bytes']
row.update({'original_local_path':str(source),'original_mode':initial.st_mode&0o777,'original_mtime_ns':initial.st_mtime_ns,'remote_host':'jkh@puck.local','seconds':time.monotonic()-start,'restore':'scp jkh@puck.local:'+remote+' . ; gzip -dc nanolang-file-service-parser-70113.retained.tar.gz > nanolang-file-service-parser-70113.tar'})
manifest=source.with_name(source.name+'.storage.json')
with manifest.open('w') as out:out.write(json.dumps(row,indent=2)+'\n');out.flush();os.fsync(out.fileno())
source.unlink()
print(json.dumps(row),flush=True)
local=source.with_name(source.name+'.gz')
subprocess.run(['scp','jkh@puck.local:'+remote,str(local)],check=True,timeout=900)
c=hashlib.sha256()
with local.open('rb') as inp:
 for b in iter(lambda:inp.read(4*1024*1024),b''):c.update(b)
assert c.hexdigest()==row['compressed_sha256']
row['local_compressed_path']=str(local);row['local_compressed_verified']=True
manifest.write_text(json.dumps(row,indent=2)+'\n')
print('Local compressed archive verified:',str(local),flush=True)
