"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import subprocess
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--repo',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='70260ec2dda340164287d33b52a977f2ff2aa8f8c3dc8c8cbfb8af81d7565c44'
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
 result=subprocess.run([__import__('sys').executable,str(root/'verify_current.py'),'--root',str(root),'--repo',args.repo],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 assert not m['qualification'] and not m['whole_release']
 for name in ['previous-full-cancelled-run.json','previous-focus-cancelled-run.json']:
  prior=json.loads((root/name).read_text());assert prior['conclusion']=='cancelled' and prior['head_sha']=='ef823c1552c6151acdd61cfda4652b49bf4220b0'
 print(result.stdout,end='')
