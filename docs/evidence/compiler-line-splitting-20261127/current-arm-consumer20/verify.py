"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='1125a4ebcfb68e10b1c34d32457bd0139e1c34c57e129a4b5b0a088b7601d879'
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
 load=lambda n:json.loads((root/n).read_text())
 x=load('observations/retained-callables-result.json')
 assert x['terminal'] and x['functional_gate_passed'] and x['all_guards_match'] and x['qualified_guards_match']
 assert (x['discovered_methods'],x['tests_run'],x['failures'],x['errors'],x['skips'])==(20,20,0,0,0)
 assert x['source_head']==m['source_head']=='c8bd5c821ac6361e6bbdef2d1750721bfaa2ba22'
 for category in ['source','tools','images']:
  before=x[category+'_before'];assert before==x[category+'_after']
  for index,(name,digest) in enumerate(sorted(before.items())):
   assert sha((root/f'observed-inputs/{category}_before/{index:05d}'/Path(name).name).read_bytes())==digest
 log=(root/'observations/suite.stderr').read_text();assert 'Ran 20 tests in ' in log and log.rstrip().endswith('OK')
 assert load('observations/queue-state.json')['suite_exit']==0
 assert not m['whole_release'] and m['qualification_execution']
print('I verified my original20-method current-Stage2 C consumer pass with zero skips and unchanged source/tool/image and qualified-root guards. Full release acceptance remains open.')
