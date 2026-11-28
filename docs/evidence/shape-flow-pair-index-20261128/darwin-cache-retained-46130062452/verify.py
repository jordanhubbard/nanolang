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
assert sha(raw)=='53dfeeacfd33ed3b058c00238646f92b9c4b157c2a25013a99b88f4363ffeb01'
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
    expected=json.loads((root/'verified-darwin-cache.json').read_text())
    result=subprocess.run([sys.executable,str(root/'verify_darwin_cache.py'),'--root',str(root),'--repo',str(args.repo),'--cohort','darwin-cache','--attempt','1'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((root/'verified-darwin-cache.json').read_text())==expected
    assert len(expected)==1
    row=expected[0]
    assert row['full_original_cache_worker_gate_passed'] and row['retention_attempt']==1
    assert (row['cache_reported_methods'],row['existing_skips'],row['worker_methods'],row['array_methods'])==(284,26,3,6)
    assert not row['whole_release']
    print('I verify the complete retained sourceDDD Darwin cache gate with original recipes and first-attempt collection. Historical missing-archive failure remains retained; full release acceptance remains open.')
