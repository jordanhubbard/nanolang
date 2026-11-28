"""I verify original gate receipts without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='cdd4704ad35991c6e83857be12ee2baf7197de83b65322e1d9672a3537a5eb23'
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
    d=load('production-tools/result.json');assert d['exit']==0 and not d['qualification']
    d=load('original-cseed-emitter/result.json')
    assert d['exit']==0 and not d['qualification'] and not d['command_timeout']
    assert d['shadow_seconds']==10 and d['outer_deadline_seconds']==300 and d['seconds']<300
    assert d['compiler_before']==d['compiler_after']=='ae09aafcfc74ee913583f23103f0e018b6af8079ff39ece5e66c02b89f532194'
    assert len(d['argv'])==4 and d['argv'][1].endswith('/tests/selfhost_shadow_emitter.nano')
    for platform in ['linux','darwin']:
        prefix='original-46063824005-'+platform+'-bootstrap.'
        job=load(prefix+'job.json')
        assert job['head_sha']=='e598013cd5363be1664a40a7fe7c615f7b2cdb2a'
        assert job['conclusion']=='failure'
        steps={s['name']:s['conclusion'] for s in job['steps']}
        assert steps['Execute the original bootstrap with its default limits']=='success'
        assert steps['Execute the original native compiler and emitter suites']=='failure'
        log=(root/(prefix+'log')).read_text()
        assert re.search(r'Z Ran 36 tests in [0-9.]+s',log)
        assert re.search(r'Z Ran 0 tests in [0-9.]+s',log)
        assert "bin/nanoc_stage2'" in log and 'timed out after 300 seconds' in log
print('I verified the local C-seed emitter pass at10/300 and the prior exact-head Linux/Darwin bootstrap/native passes followed by original Stage2 timeouts. Full current platform/release qualification remains open.')
