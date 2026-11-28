"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import subprocess
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--repo',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='24448cf46cb94171b0ee13648bb894d9b74a8c5f16a7eb6c97e37f49147c7d62'
m=json.loads(raw)
with tempfile.TemporaryDirectory(prefix='nano-arm-receipt-',dir=args.workspace) as d:
 root=Path(d)
 for row in m['files']:
  name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts
  output=root/name;output.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with output.open('wb') as stream:
   for key in row['chunks']:
    packed=(b/'objects'/(key+'.gz')).read_bytes();meta=m['chunks'][key]
    assert sha(packed)==meta['gzip_sha256'];data=gzip.decompress(packed)
    assert len(data)==meta['bytes'] and sha(data)==key
    stream.write(data);h.update(data);size+=len(data)
  assert size==row['size'] and h.hexdigest()==row['sha256']
 result=subprocess.run([__import__('sys').executable,str(root/'verify_baseline.py'),'--root',str(root),'--repo',args.repo],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 assert not m['qualification'] and not m['whole_release']
 print(result.stdout,end='')
