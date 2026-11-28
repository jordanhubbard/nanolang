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
assert sha(raw)=='22f2a9a617f07d6dad8796bf8ac4451d154ccf77d385997903a6e7d006d71e80'
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
    observed=json.loads((root/'verified-cohorts.json').read_text())
    result=subprocess.run([sys.executable,str(Path(__file__).with_name('verify_original_cohort.py')),'--root',str(root),'--repo',str(args.repo)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert observed==json.loads((root/'verified-cohorts.json').read_text())
    print(result.stdout,end='')
print('I verified complete original Linux/Darwin17/36/12 and Linux284-cache/3-worker/6-array cohorts, exact whole ZIPs and retained source/tool/image/host identities. Darwin cache and full release remain open.')
