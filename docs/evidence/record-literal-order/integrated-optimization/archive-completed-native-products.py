from pathlib import Path
import hashlib,json,os,shutil,tarfile,time
summary=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
for d in sorted(Path('/tmp').glob('nano-native-literal-order-*')):
 p=d/'commands.json'
 if not p.exists():continue
 rows=json.loads(p.read_text())
 if len(rows)<18 or any(r.get('returncode')!=0 for r in rows):continue
 paths={Path(r['command'][0]) for r in rows if r.get('stage')=='run'}
 if not paths or not all(p.parent==d and p.is_file() and not p.is_symlink() for p in paths):continue
 roots={r['command'][0] for r in rows if r.get('stage')=='compile'}
 if not all('/nanolang-record-literal-source-order/bin/' in s or '/nanolang-literal-order-' in s for s in roots):continue
 archive=d/'successful-products.tar.gz';temporary=d/'successful-products.tar.gz.partial'
 assert not archive.exists() and not temporary.exists()
 inventory={p.name:{'sha256':digest(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(paths)}
 with tarfile.open(temporary,'w:gz',compresslevel=6) as tf:
  for p in sorted(paths):tf.add(p,arcname=p.name,recursive=False)
 with tarfile.open(temporary,'r:gz') as tf:
  assert len(tf.getmembers())==len(inventory)
  for member in tf:
   row=inventory[member.name];assert member.isfile() and member.size==row['bytes'] and member.mode==row['mode']
   assert hashlib.sha256(tf.extractfile(member).read()).hexdigest()==row['sha256']
 for p in paths:
  row=inventory[p.name];assert digest(p)==row['sha256'] and p.stat().st_mtime_ns==row['mtime_ns'] and p.stat().st_mode&0o777==row['mode']
 temporary.rename(archive)
 record={'archive':str(archive),'archive_sha256':digest(archive),'members':inventory,'verified_exact_before_removing_uncompressed_copies':True,'commands_sha256':digest(d/'commands.json')}
 (d/'successful-products-archive.json').write_text(json.dumps(record,indent=2)+'\n')
 (d/'RESTORE_PRODUCTS.txt').write_text('I retain the exact successful executables in successful-products.tar.gz. Each member was rehashed and its mode checked before its uncompressed copy was removed. Restore in this directory with: tar -xzf successful-products.tar.gz\n')
 for p in paths:p.unlink()
 saved=sum(r['bytes'] for r in inventory.values())-archive.stat().st_size
 summary.append({'directory':str(d),'members':len(inventory),'reclaimed_bytes':saved,'archive_sha256':record['archive_sha256']})
 print(d,saved,flush=True)
Path('/tmp/nanolang-native-product-archive-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Total reclaimed',sum(r['reclaimed_bytes'] for r in summary),'free',shutil.disk_usage('/tmp').free,flush=True)
