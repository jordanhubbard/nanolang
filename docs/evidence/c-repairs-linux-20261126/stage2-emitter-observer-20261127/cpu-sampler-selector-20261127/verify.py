"""I verify my CPU sampler controls without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='900895990d806ac3adace56d0822b746587c7d8d12843e0eb6d566c2db8f8e09'
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
    load=lambda name:json.loads((root/name).read_text())
    job=load('hosted-46061588260-job.json')
    assert (job['id'],job['run_id'],job['head_sha'],job['conclusion'])==(140716888391,46061588260,'7f794456a9babbdede62528e3fda778049325269','failure')
    steps={s['name']:s for s in job['steps']}
    assert steps['Provision diagnostic CPU sampling']['conclusion']=='failure'
    assert steps['Observe the retained command at its original limit']['conclusion']=='skipped'
    zpath=root/'hosted-46061588260-artifact.zip'
    assert sha(zpath.read_bytes())=='d3cfd8d6369c41eaad3ec5ddd92b02696c60d180134a7960279252856ec0231a'
    with zipfile.ZipFile(zpath) as z:
        assert len(z.namelist())==2 and z.testzip() is None
        assert z.read('stage2-emitter-controls/checkout-revision.txt').strip()==b'd18fed86140b6a4cf983898ac32a079002f6b784'
    d=load('actual-elf-control/sample/record.json')
    assert d['valid'] and d['event']=='armv8_pmuv3_1/cycles/u'
    assert d['exit_code']==-2 and d['interrupt_sent'] and not d.get('forced_stop')
    assert d['perf_sha256_before']==d['perf_sha256_after']
    assert '/linux-nvidia-' in d['perf']
    assert sha((root/'actual-elf-control/sample/perf.data').read_bytes())==d['data_sha256']
    script=(root/'actual-elf-control/sample/script.stdout').read_text()
    count=len(re.findall(r'(?m)^.*\b\d+\.\d+:\s+\d+\s+\S+:\s*$',script))
    assert count==d['actual_samples']==95 and 'nano_stage2_sample_control' in script
    provision=load('selector-control-corrected/.test_output/stage2-emitter-controls/perf-provision.json')
    selected=provision['binaries'][-1]
    assert selected['path']==d['perf'] and selected['sha256']==d['perf_sha256_before']
    c=load('actual-elf-control/control.json')
    assert c['valid'] and c['before']==c['after'] and c['sample_count']==95
    assert (root/'pinned-protocol-21.stderr').read_text().rstrip().endswith('OK')
    assert 'Ran 21 tests in ' in (root/'pinned-protocol-21.stderr').read_text()
    workflow=(root/'source/.github/workflows/stage2-emitter-diagnostic.yml').read_text()
    assert "['dpkg-query','-S','*/perf']" in workflow
    assert '--sample-cpu' in workflow
print('I verified the hosted setup refusal, actual ELF selector,95 native calibration samples and21 protocol controls; hosted CPU evidence remains pending.')
