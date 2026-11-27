"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='13a55686c556db8e5c49f64e1506bfe01f8fe92105bbee338a0c1ce4b1d6bb91'
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
 final=load('observations/final-focused/result.json');assert final['exit']==0 and final['before']==final['after'] and not final['qualification']
 text=(root/'observations/final-focused/stderr').read_text();assert 'Ran 13 tests' in text and text.rstrip().endswith('OK')
 prior=load('observations/prior-instrumented/result.json');assert prior['exit']==1 and prior['before']==prior['after']
 assert '18236773 not less than 446656' in (root/'observations/prior-instrumented/stderr').read_text()
 negative=load('observations/final-prior-control/result.json');assert negative['expected_refusal'] and negative['commands'][-1]['exit']!=0
 assert 'heap-use-after-free' in (root/'observations/final-prior-control/2.stderr').read_text()
 assert negative['final_test_sha256']==sha((root/'source-and-products/tests/test_eval_string_split.c').read_bytes())
 for name in ['split-repaired-normal.log','split-repaired-asan.log']:
  log=(root/'observations'/name).read_text();assert log.count('I retained ')==9 and 'ERROR: AddressSanitizer' not in log and 'runtime error:' not in log
 results=list((root/'focused').glob('*/length-work.json'));assert len(results)==1
 metrics=json.loads(results[0].read_text());assert metrics=={'input_bytes':5955,'observed':{'prior':{'calls':6123,'requested_bytes':18236773},'current':{'calls':28,'requested_bytes':5421}}}
 assert load('observations/workflow-parse.json')['system_python_workflow_yaml_parse']
 assert not m['whole_release'] and not m['full_qualification']
print('I verified my exact prior/five-route/length-work regression, nine normal/sanitized lifetime cases and old-code ASan refusal; full qualification remains open.')
