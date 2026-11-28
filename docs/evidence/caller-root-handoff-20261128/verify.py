"""I verify my scoped compiler splitter and C lifetime repair without executing products."""
from pathlib import Path
import argparse,gzip,hashlib,json,tempfile,subprocess
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--repo',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='528a91275d095f2a83cf4b46590a11f5467bab0c67b0f0f7e096a5d16fe0dc44'
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
 assert not m['qualification'] and m['qualified_private_component'] and not m['whole_release'] and not m['default_activation']
 q=load('observations/qualification-result.json')
 assert q['terminal'] and q['guards_match'] and q['source_before']==q['source_after'] and q['source_base']==m['source_base']
 for name in m['source_names']:
  data=(root/'source'/name).read_bytes()
  assert sha(data)==q['source_before'][name]
  assert data==subprocess.check_output(['git','show',m['source_commit']+':'+name],cwd=args.repo)
 original=(root/'source/tests/support/ephemeral_array_cases.inc').read_bytes().replace(b'../../src/',b'../src/')
 identity=load('observations/original-case-identity.json')
 assert identity['all_assertions_and_fixtures_unchanged'] and sha(original)==identity['sha256']=='68841685216b26afea88f974dba48e5f2e630935f652b52a28fafe29e17ca61f'
 assert original==subprocess.check_output(['git','show',identity['upstream_commit']+':'+identity['upstream_path']],cwd=args.repo)
 test=(root/'source/tests/test_eval_caller_root_handoff.c').read_text()
 assert 'env_carrier_export(' not in test
 assert [run['mode'] for run in q['runs']]==['normal','sanitized'] and all(run['returncode']==0 for run in q['runs'])
 assert q['runs'][1]['sanitizer_environment']=={'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'}
 for mode in ['normal','sanitized']:
  log=(root/f'observations/{mode}-accepted.log').read_text()
  rows=[json.loads(line) for line in log.splitlines() if line.startswith('{"case":')]
  assert [row['expected_reclaims'] for row in rows]==[2,2,5,2,65,2,2,2,1,1]
  assert all(row['expected_reclaims']==row['actual_reclaims'] for row in rows)
  assert 'I preserve custody declarations through 6 allocation failures and recover.' in log
  assert 'I refuse owned record graphs while scalar list cells and borrow-only keepalive roots grant no custody handoff.' in log
  assert 'I preserve overlapping cyclic caller roots without adopting storage.' in log
  for text in ['ERROR: AddressSanitizer','ERROR: LeakSanitizer','runtime error:','make: ***']:assert text not in log
 mutation=load('observations/mutation-result.json')
 assert mutation['returncode']==-6 and mutation['unchanged_reclaim_guard_refused'] and not mutation['whole_release']
 assert mutation['test_source_sha256']==sha(test.encode()) and mutation['case_source_sha256']==sha((root/'source/tests/support/ephemeral_array_cases.inc').read_bytes())
 assert 'env_static_array_reclaim_count() == before' in (root/'observations/mutation-accepted.stderr').read_text()
 source=(root/'source/src/eval.c').read_text()
 assert (root/'observations/mutant-eval-return-only.c').read_text()==source.replace('env_carrier_export_call(env, value, list, !borrowed_get);','if (!borrowed_get) env_carrier_export(env, value, list);')
 assert 'SUMMARY: AddressSanitizer: 7 byte(s) leaked in 7 allocation(s).' in (root/'observations/first-fixture-leak.log').read_text()
print('I verified my private caller-root handoff, all ten unchanged cases, six allocation failures, normal/sanitizer neighbors and the old-handoff mutation refusal. Default ownership and release acceptance remain open.')
