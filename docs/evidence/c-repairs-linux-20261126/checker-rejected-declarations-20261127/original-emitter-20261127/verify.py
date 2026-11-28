"""I verify my retained C dispatch controls without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='cbf2f46c2deb4c8db9463c45d1b58d97e5073bf601ab524f57b02f73b7861ade'
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
    d=json.loads((root/'result.json').read_text())
    assert not d['qualification'] and d['exit']==0 and not d['command_timeout']
    assert d['shadow_seconds']==10 and d['outer_deadline_seconds']==300 and d['seconds']<300
    assert d['guards_match'] and d['source_before']==d['source_after']
    ids=json.loads((root/'provider-identities.json').read_text())
    assert d['compiler_before']==d['compiler_after']==ids['candidate_compiler']['sha256']
    assert len(d['argv'])==4 and d['argv'][1].endswith('/tests/selfhost_shadow_emitter.nano')
print('I verified the local corrected-C emitter pass at the original10/300-second limits; full platform qualification remains open.')
