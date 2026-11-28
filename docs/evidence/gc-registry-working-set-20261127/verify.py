"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='e3d70e6a510fc711c6ded1b2a5344ad6070c4a4a72ed96dfed7dadb8ec4f9964'
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
 assert not m['qualification'] and not m['whole_release']
 s=load('focused/focused-summary.json');assert s['normal_and_sanitizers'] and s['original_tests_unchanged'] and s['source_and_tool_guards_match']
 assert not s['speedup_established'] and not s['original_cause_established']
 for name,digest in s['source_sha256'].items():assert sha((root/'final-source'/name).read_bytes())==digest
 log=(root/'focused/final-fresh-boundaries-corrected.log').read_text()
 assert log.count('I preserved both essential allocation failures around a ')==8
 assert 'runtime error:' not in log and 'ERROR: AddressSanitizer' not in log
 assert 'within_budget":false' in (root/'focused/baseline.log').read_text()
 assert load('focused/bad-rehash-control/result.json')['mutant_exit']!=0
 assert 'gc_is_managed(owner)' in (root/'focused/bad-rehash-control/result.log').read_text()
 for label,path in [('candidate','focused/original-fixture'),('profile','profile/diagnostic')]:
  before=load(path+'/identities-before.json');assert before==load(path+'/identities-after.json')
  for index,(name,digest) in enumerate(sorted(before.items())):
   assert sha((root/f'observed-inputs/{label}/{index:05d}'/Path(name).name).read_bytes())==digest
  d=load(path+'/diagnostic.json');p=d['product'];assert p['terminal'] and p['exit_code']==0 and not p['command_timeout'] and p['deadline_seconds']==300
  assert not p['observer_errors'] and p['output_complete']
  lines=(root/path/'command/stderr').read_text().splitlines();events=[json.loads(x) for x in lines if x.startswith('{')]
  status=[x for x in events if x.get('phase')=='supervisor-status'];assert len(status)==1
  u=status[0];assert u['completed'] and u['reaped'] and u['exit_code']==0 and u['deadline_seconds']==10 and not u['parent_timeout'] and not u['clock_failure']
 p=load('profile/diagnostic/diagnostic.json')['product']['cpu_profile'];assert p['valid'] and p['actual_samples']==3358
 assert sha((root/'profile/diagnostic/command/cpu-profile/perf.data').read_bytes())==p['data_sha256']
 c=load('profile/corrected-child-self-analysis.json');assert c['child_samples']==330 and c['child_period']==14620811484
 assert next(x for x in c['rows'] if x['symbol']=='gc_alloc')['period']==2629192447
 assert load('profile/calibration/control.json')['valid']
print('I verified scoped GC working-set controls and the Linux ARM counterpart profile. I have not established a compiler speedup, the Darwin cause, full qualification or release acceptance.')
