"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import subprocess
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--repo',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='bc753622c0606def77b566a8a067956b242035a3995c4803013c8546fe4d583a'
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
 assert x['source_head']==m['source_head']=='f8e7b78334281485c3b48765c835b8bda218c975'
 assert x['prerequisite_current_stage2'] and not x['prerequisite_reconstructed']
 assert x['prerequisite_receipt_sha256']=='2cce974db50173897ba3db2ea331793a5a005fcba3e98b0fce8bfa4a456b0bcd'
 assert x['prerequisite_native_sha256']=='a09a6396b59dc2b61a2a28f71f45bf08a4af41bcf0c83f18529ff5bd6b2d60b4'
 compiler=[n for n in x['tools_before'] if n.endswith('/root-slot-native-20261128/bin/nanoc_stage2')];assert len(compiler)==1
 assert x['tools_before'][compiler[0]]=='8ea017747c8e195ec64a4486d9f7ffcde93a7e478e0ae6ef2eb0bcd819107913'
 assert any(command['argv'][0]==compiler[0] for command in x['commands'])
 for name in ['tests/test_nanolang_translator_affine_analysis.py','tests/test_nanolang_translator_affine_public.py','tests/test_nanolang_translator_affine_analysis_cache.py','tests/test_nanolang_translator_shape_solver.py','tests/test_nanolang_translator_callables.py','tests/test_native_nonscalar_cast.py','tests/test_native_array_write_inference.py','tests/test_native_record_pointer_storage.py','tests/native_process.py','tests/nanolang_translator_routes.py']:
  assert subprocess.check_output(['git','show','c4264e3e7e9f4c1fde0b8f4c12068981b40ca225:'+name],cwd=args.repo)==subprocess.check_output(['git','show',m['source_head']+':'+name],cwd=args.repo)

 for category in ['source','tools','images']:
  before=x[category+'_before'];assert before==x[category+'_after']
  for index,(name,digest) in enumerate(sorted(before.items())):
   assert sha((root/f'observed-inputs/{category}_before/{index:05d}'/Path(name).name).read_bytes())==digest
 log=(root/'observations/suite.stderr').read_text();assert 'Ran 20 tests in ' in log and log.rstrip().endswith('OK')
 assert load('observations/queue-state.json')['suite_exit']==0
 for command in x['commands']:
  for stream in ['stdout','stderr']:
   if stream in command:
    observed=Path(command[stream]['path'])
    path=root/'observations'/observed.relative_to('/home/jkh/.local/share/fleet-migration/puck-volume/nanolang/tmp/root-slot-consumers-20261128')
    assert sha(path.read_bytes())==command[stream]['sha256']
 assert load('worker-admission/controller-result.json')['exit']==0
 assert load('worker-admission/lease-watchdog-completed.json')['owned_controllers_terminal']
 assert load('worker-admission/worker-reservation-release.json')['status']=='released'
 assert load('worker-admission/worker-reservation-release.json')['fence']==28
 assert not m['whole_release'] and m['qualification_execution']
print('I verified my original20-method current-Stage2 C consumer pass with zero skips and unchanged source/tool/image and qualified-root guards. Full release acceptance remains open.')
