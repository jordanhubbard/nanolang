"""I verify my CPU sampler controls without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='9cec98cb2c76e1921fbabe9793047636cfda1e9c8972dcd4c5fe9e591a12b85e'
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
    load=lambda name:json.loads((root/name).read_text())
    old=load('native-control/sample/record.json')
    assert old['exit_code']==255
    assert "PMU Hardware doesn't support sampling/overflow-interrupts" in (root/'native-control/sample/record.stderr').read_text()
    initial=load('hardware-control/sample/record.json')
    assert initial['exit_code']==-2 and not initial.get('valid',False)
    d=load('hardware-control-corrected/sample/record.json')
    assert d['valid'] and d['event']=='armv8_pmuv3_1/cycles/u'
    assert d['exit_code']==-2 and d['interrupt_sent'] and not d.get('forced_stop')
    assert d['perf_sha256_before']==d['perf_sha256_after']
    assert sha((root/'hardware-control-corrected/sample/perf.data').read_bytes())==d['data_sha256']
    script=(root/'hardware-control-corrected/sample/script.stdout').read_text()
    count=len(re.findall(r'(?m)^.*\b\d+\.\d+:\s+\d+\s+\S+:\s*$',script))
    assert count==d['actual_samples']==92
    assert 'nano_stage2_sample_control' in script
    for name in ['script','report','buildids']:
        row=d[name]
        assert row['exit_code']==0
        assert sha((root/('hardware-control-corrected/sample/'+name+'.stdout')).read_bytes())==row['stdout_sha256']
        assert sha((root/('hardware-control-corrected/sample/'+name+'.stderr')).read_bytes())==row['stderr_sha256']
    c=load('hardware-control-corrected/control.json')
    assert c['valid'] and c['before']==c['after'] and c['sample_count']==92
    log=(root/'pinned-protocol.stderr').read_text()
    assert 'Ran 20 tests in ' in log and log.rstrip().endswith('OK')
    workflow=(root/'source/.github/workflows/stage2-emitter-diagnostic.yml').read_text()
    assert '--sample-cpu' in workflow and 'kernel.perf_event_paranoid=2' in workflow
print('I verified actual ARM sampler calibration and20 protocol controls; hosted software-event profiling and qualification remain open.')
