"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import subprocess
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--repo',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='74b65e57ce0dc72ce1f7b1957c08ea684e2be27a8ca7f25da55e688b4abb3577'
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
 result=subprocess.run([__import__('sys').executable,str(root/'verify_failure.py'),'--root',str(root),'--repo',args.repo],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 assert not m['qualification'] and not m['whole_release']
 assert 'Ran 5 tests in ' in (root/'policy-controls/timeout-policy.stderr').read_text()
 assert (root/'policy-controls/timeout-policy.stderr').read_text().rstrip().endswith('OK')
 negative=(root/'policy-controls/old-global-override.stderr').read_text()
 assert 'inherited override: NANO_SHADOW_TIMEOUT_SECONDS' in negative and 'FAILED' in negative
 for p in (root/'corrected-observer').rglob('*'):
  if p.is_file():
   name=str(p.relative_to(root/'corrected-observer'))
   assert p.read_bytes()==subprocess.check_output(['git','show',m['corrected_observer']+':'+name],cwd=args.repo)
 print(result.stdout,end='')
