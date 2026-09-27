from pathlib import Path
import hashlib,json,os,subprocess
manifest=Path('/tmp/nanolang-file-service-parser-70113.tar.storage.json');row=json.loads(manifest.read_text())
p=Path('/tmp/nanolang-file-service-parser-final-artifacts')/row['original_sha256'];st=p.stat()
assert p.is_file() and not p.is_symlink() and st.st_nlink==1 and st.st_size==row['original_bytes']
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==row['original_sha256']
remote=subprocess.check_output(['ssh','jkh@puck.local','/usr/bin/shasum -a 256 '+row['remote_path']],text=True)
assert remote.split()[0]==row['compressed_sha256']
packed=p.with_name(p.name+'.gz');index=p.parent/'packed-object-70113.json'
entry={'raw_path':str(p),'sha256':row['original_sha256'],'bytes':row['original_bytes'],'mode':st.st_mode&0o777,'mtime_ns':st.st_mtime_ns,'packed_path':str(packed),'packed_sha256':row['compressed_sha256'],'packed_bytes':row['compressed_bytes'],'verified_remote_host':row['remote_host'],'verified_remote_path':row['remote_path'],'restore_command':'gzip -dc '+str(packed)+' > '+str(p)}
with index.open('w') as out:out.write(json.dumps(entry,indent=2)+'\n');out.flush();os.fsync(out.fileno())
assert p.stat().st_ino==st.st_ino and p.stat().st_mtime_ns==st.st_mtime_ns
p.unlink()
subprocess.run(['scp',row['remote_host']+':'+row['remote_path'],str(packed)],check=True,timeout=900)
h=hashlib.sha256()
with packed.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()==row['compressed_sha256']
entry['local_packed_copy_verified']=True;index.write_text(json.dumps(entry,indent=2)+'\n')
row['packed_cas_manifest']=str(index);row['local_compressed_path']=str(packed);row['local_compressed_verified']=True
row.pop('local_compressed_copy_removed',None)
manifest.write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps(entry),flush=True)
