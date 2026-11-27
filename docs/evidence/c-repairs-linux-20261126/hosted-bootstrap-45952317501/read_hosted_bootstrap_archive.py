"""I check retained hosted bytes and preserve the recorded qualification gaps."""
import argparse,gzip,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=(a.archive/'manifest.json').read_bytes();assert sha(raw)=='743197d205c6f5dbb85a0e674e8ea5d3ac61f643006f90c385d1b497ed458b8d';q=json.loads(raw)
assert q['run']==45952317501 and q['artifact']==15178888555
assert q['revision']=='ee5d38a4b401249ecb7dcfaa0b7b04001fdc8dc1'
seen=set();used=set()
with tempfile.TemporaryDirectory(prefix='hosted-bootstrap-reader-',dir=a.workspace) as temp:
 root=Path(temp)
 for r in q['files']:
  rel=Path(r['path']);assert not rel.is_absolute() and '..' not in rel.parts
  assert r['path'] not in seen;seen.add(r['path'])
  f=root/rel;f.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with f.open('wb') as out:
   for key in r['chunks']:
    packed=(a.archive/'objects'/(key+'.gz')).read_bytes();meta=q['chunks'][key]
    assert sha(packed)==meta['gzip_sha256'];block=gzip.decompress(packed)
    assert len(block)==meta['bytes'] and sha(block)==key
    out.write(block);h.update(block);size+=len(block);used.add(key)
  assert size==r['size'] and h.hexdigest()==r['sha256'],r['path']
 assert used==set(q['chunks'])
 result=subprocess.run([sys.executable,str(Path(__file__).with_name('check_hosted_bootstrap.py')),'--artifact',str(root/q['root']),'--repo',str(a.repo)],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 print(result.stdout,end='')
 print(json.dumps({'archive_records_verified':len(seen),'archive_chunks_verified':len(used),'manifest_sha256':sha(raw),'generated_index_bytes_retained':False,'whole_release':False}))
