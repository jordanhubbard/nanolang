"""I verify my original native cohort archives without executing a product."""
import subprocess,sys
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--repo',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='c0a6328f16fc963420bd718be6ca391cbd0356f9583813b49aea23d88d45a9ef'
manifest=json.loads(raw)
assert not manifest['qualification'] and not manifest['whole_release']
with tempfile.TemporaryDirectory(prefix='namespace-admission-reader-', dir=args.workspace) as temporary:
    root = Path(temporary)
    used, seen = set(), set()
    for record in manifest['files']:
        relative = Path(record['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert str(relative) not in seen
        seen.add(str(relative))
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with target.open('wb') as stream:
            for key in record['chunks']:
                assert re.fullmatch('[a-f0-9]{64}', key)
                packed = (args.archive / 'objects' / (key + '.gz')).read_bytes()
                metadata = manifest['chunks'][key]
                assert sha(packed) == metadata['gzip_sha256']
                block = gzip.decompress(packed)
                assert sha(block) == key and len(block) == metadata['bytes']
                digest.update(block)
                size += len(block)
                stream.write(block)
                used.add(key)
        assert size == record['size'] and digest.hexdigest() == record['sha256']
    assert used == set(manifest['chunks'])
    m=manifest
    load=lambda name:json.loads((root/name).read_text())
    assert not m['qualification'] and not m['whole_release']
    observed=load('failed-cohorts/verified-linux-refusals.json')
    result=subprocess.run([sys.executable,str(root/'failed-cohorts/verify_linux_refusals.py'),'--root',str(root/'failed-cohorts'),'--repo',str(args.repo)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert observed==load('failed-cohorts/verified-linux-refusals.json')
    assert len(observed)==2 and all(not r['bootstrap_native_cache_acceptance'] and not r['caller_regression_execution'] and not r['whole_release'] for r in observed)
    for revision,label in [(m['old_source'],'old'),(m['new_source'],'new')]:
        for name in m['source_names']:
            assert (root/'source'/label/name).read_bytes()==subprocess.check_output(['git','show',revision+':'+name],cwd=args.repo)
    names=subprocess.check_output(['git','diff','--name-only',m['old_source'],m['new_source']],cwd=args.repo,text=True).splitlines()
    assert names==['docs/ROADMAP.md','tests/support/ephemeral_array_cases.inc']
    old=(root/'source/old/tests/support/ephemeral_array_cases.inc').read_text()
    new=(root/'source/new/tests/support/ephemeral_array_cases.inc').read_text()
    guard='#ifndef _POSIX_C_SOURCE\n#define _POSIX_C_SOURCE 200809L\n#endif\n#ifndef _DARWIN_C_SOURCE\n#define _DARWIN_C_SOURCE 1\n#endif\n'
    original='#define _POSIX_C_SOURCE 200809L\n#define _DARWIN_C_SOURCE 1\n'
    assert new==old.replace(original,guard,1)
    assert sha(old.replace('../../src/','../src/').encode())=='68841685216b26afea88f974dba48e5f2e630935f652b52a28fafe29e17ca61f'
    control=load('controls/control-result.json');assert not control['qualification'] and not control['whole_release'] and control['assertions_fixtures_and_body_unchanged']
    assert control['guarded_include_sha256']==sha(new.encode()) and control['undef_warning_suppression_mutation_refused'] and control['selected_macro_value_assertion']==202405
    assert [r['exit'] for r in control['rows']]==[1,0]
    assert [r['exit'] for r in control['preservation_and_mutation_rows']]==[1,0,1]
    assert all('-Werror' in r['argv'] and '-fsyntax-only' in r['argv'] for r in control['rows']+control['preservation_and_mutation_rows'])
    assert (root/'controls/undef-mutation.inc').read_text()=='#undef _POSIX_C_SOURCE\n'+old
    assert '#error I require the selected POSIX macro value' in (root/'controls/undef-mutation.stderr').read_text()
    assert not (root/'controls/new-guarded-with-preservation.stderr').read_bytes()
    print(result.stdout,end='')
print('I verified both original Linux macro refusals, exact guarded-only source delta, unchanged cases and selected-value/undef-mutation controls. Fresh complete qualification remains required.')
