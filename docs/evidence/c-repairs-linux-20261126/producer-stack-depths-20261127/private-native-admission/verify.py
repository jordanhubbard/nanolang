"""I verify my private native-link admission without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='42430903a4a9ace88803107314087b995ca24ef2080ca448f18e0456c6a4fb3e'
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
    d=load('diagnostic/result.json');assert not d['qualification'] and not d['whole_release']
    assert d['exit']==0 and not d['command_timeout'] and d['deadline_seconds']==1800
    assert d['guards_match'] and d['before']==d['after'] and len(d['before'])==5
    assert d['original_step']['label']=='stage1-native' and d['original_step']['exit_code']==1
    argv=d['argv'][:];argv[argv.index('-o')+1]=d['original_step']['argv'][d['original_step']['argv'].index('-o')+1];assert argv==d['original_step']['argv']
    assert '/puck-volume/' in d['TMPDIR'] and d['TMPDIR'].endswith('/private-native-link/cc-tmp')
    assert d['native_sha256']==sha((root/'diagnostic/stage1').read_bytes())=='e1274efe6e2a2d9fdd517629cb39575afa20649d5da90459fe3243dbf2f28243'
    assert (root/'diagnostic/stage1').read_bytes()[:4]==b'\x7fELF'
    assert (root/'diagnostic/stdout').read_bytes()==(root/'diagnostic/stderr').read_bytes()==b''
    admission=load('full-qualification-admission.json');assert admission['qualification'] and not admission['whole_release']
    assert admission['source_head']=='6bae061d14f2559856aedb422f552c861e57ae95'
    assert admission['admission_receipt_sha256']==sha((root/'diagnostic/result.json').read_bytes())
    assert admission['admission_native_sha256']==d['native_sha256'] and admission['TMPDIR']!=d['TMPDIR']
print('I verified the exact retained native input links at1800 with private TMPDIR and unchanged guards, and separately bound the fresh full-run admission. No full bootstrap/platform/release result is asserted.')
