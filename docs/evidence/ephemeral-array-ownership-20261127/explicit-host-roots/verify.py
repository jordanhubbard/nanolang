"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import argparse,gzip,hashlib,json,tempfile,subprocess
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--repo',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='b1d78877fe0ecda16d5b4b7d4a0529d205e61b8355c5f4305b71ba3399184234'
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
 assert not m['qualification'] and not m['whole_release'] and not m['default_or_product_mode_changed']
 for name in m['source_names']:
  assert (root/'source'/name).read_bytes()==subprocess.check_output(['git','show',m['parent_commit']+':'+name],cwd=args.repo)
 for i,name in enumerate(m['parent_manifest_names']):
  assert (root/'parents'/f'{i}.json').read_bytes()==subprocess.check_output(['git','show',m['parent_commit']+':'+name],cwd=args.repo)
 original=(root/'source/tests/test_eval_ephemeral_arrays.c').read_text()
 expected=original.replace('#include "../src/nanolang.h"','#include "../src/nanolang.h"\n#include "../src/eval/carrier_runtime.h"').replace('    assert(type_check(f.program, f.env));','    assert(type_check(f.program, f.env));\n    assert(env_carrier_enable(f.env));')
 provider=load('provider/receipt.json');assert sha(original.encode())==provider['test_sha256']
 assert provider['ASAN_OPTIONS']=='detect_leaks=1:halt_on_error=1' and provider['UBSAN_OPTIONS']=='halt_on_error=1'
 for name in ['explicit-carrier-admission-control','explicit-host-root-control']:
  q=load(name+'/result.json');assert q['before']==q['after'] and q['guards_match'] and not q['qualification'] and not q['default_mode_changed'] and not q['product_source_changed']
  assert q['provider_receipt_sha256']==sha((root/'provider/receipt.json').read_bytes())
  assert all(q['before'][k]==v for k,v in provider['inputs_before'].items())
  assert q['operations'][0]['exit']==0 and '-fsanitize=address,undefined' in q['operations'][0]['argv']
  prototype=(root/name/'test_explicit_carrier.c').read_text()
  if name=='explicit-carrier-admission-control':
   assert prototype==expected and q['operations'][1]['exit']==-6
   assert 'env_static_array_reclaim_count() == before' in (root/name/'run.stderr').read_text()
   assert (root/name/'run.stdout').read_text()=='I preserve caller-supplied and C-caller-returned array storage.\n'
  else:
   marker='    assert(result.type == VAL_INT && result.as.int_val == 7);'
   assert expected.count(marker)==2
   assert prototype==expected.replace(marker,marker+'\n    env_carrier_export(f.env, outer, false);')
   assert q['operations'][1]['exit']==0 and not (root/name/'run.stderr').read_bytes()
   rows=[json.loads(line) for line in (root/name/'run.stdout').read_text().splitlines() if line.startswith('{')]
   assert len(rows)==10
   assert [row['expected_reclaims'] for row in rows]==[2,2,5,2,65,2,2,2,1,1]
   assert all(row['expected_reclaims']==row['actual_reclaims'] for row in rows)
   assert 'I found 0 temporary-array ownership failures.' in (root/name/'run.stdout').read_text()
print('I verified the explicit-activation refusal and all ten sanitizer/LSan ownership controls with valid manual caller-root handoff. Default behavior and automatic public-boundary adaptation remain unimplemented.')
