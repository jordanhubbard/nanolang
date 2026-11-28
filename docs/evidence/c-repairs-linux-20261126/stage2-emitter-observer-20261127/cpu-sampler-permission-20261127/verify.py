"""I verify my CPU sampler controls without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='c7eaba7ccdb39e353959ffd588534d89ca28478d311089d6c5c35b1aa6a1710c'
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
    import zipfile
    job=json.loads((root/'job.json').read_text())
    assert (job['id'],job['run_id'],job['head_sha'],job['conclusion'])==(140718988253,46062228728,'4683a1ccc3946a54f30f2142efc04dfd364f730a','failure')
    steps={s['name']:s for s in job['steps']}
    assert steps['Provision diagnostic CPU sampling']['conclusion']=='success'
    assert steps['Calibrate process observation and cache restoration']['conclusion']=='success'
    assert steps['Observe the retained command at its original limit']['conclusion']=='failure'
    assert sha((root/'artifact.zip').read_bytes())=='c76d5079689255528fa531a55500f948ffe98d9218e7425ebc0312fc7a56acfd'
    with zipfile.ZipFile(root/'artifact.zip') as z:
        assert len(z.namelist())==35 and z.testzip() is None
        load=lambda name:json.loads(z.read('stage2-emitter-diagnostic/'+name))
        p=load('cpu-sampler-control/sample/record.json')
        assert p['exit_code']==255 and p['event']=='cpu-clock:u' and not p.get('valid',False)
        assert p['perf_sha256_before']==p['perf_sha256_after']=='c6c3c8573dfafe0592431078a87e88639029c3ec15e3300aae00f52e37547831'
        error=z.read('stage2-emitter-diagnostic/cpu-sampler-control/sample/record.stderr').decode()
        assert 'Access to performance monitoring' in error and 'perf_event_paranoid setting is 2' in error
        assert 'Ran 21 tests in ' in z.read('stage2-emitter-controls/stderr').decode()
        assert 'stage2-emitter-diagnostic/command/result.json' not in z.namelist()
        assert 'stage2-emitter-diagnostic/identities-before.json' not in z.namelist()
        provision=json.loads(z.read('stage2-emitter-controls/perf-provision.json'))
        assert any(row['sha256']==p['perf_sha256_before'] for row in provision['binaries'])
    old=(root/'original-workflow.yml').read_text();new=(root/'corrected-workflow.yml').read_text()
    assert 'kernel.perf_event_paranoid=2' in old
    assert 'kernel.perf_event_paranoid=-1' in new
    assert "'diagnostic_worker_only':True" in new and "'perf_event_paranoid':Path('/proc/sys/kernel/perf_event_paranoid')" in new
    for text in [old,new]:
        assert '--sample-cpu' in text and 'd18fed86140b6a4cf983898ac32a079002f6b784' in text
print('I verified the real hosted software-event permission refusal before product execution; corrected calibration and qualification remain pending.')
