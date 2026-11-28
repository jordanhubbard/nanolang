"""I verify my current Linux cache cohort without executing a product."""
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
assert sha(raw)=='e5417a56125d06a6672836aa546e88cfeb8c1854b7dc093e1c986bc11d824698'
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
    observed=json.loads((root/'verified-linux-cache.json').read_text())
    result=subprocess.run([sys.executable,str(root/'verify_linux_cache.py'),'--root',str(root),'--repo',str(args.repo)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert observed==json.loads((root/'verified-linux-cache.json').read_text())
    assert len(observed)==1 and observed[0]['existing_skips']==34 and observed[0]['direct_split_lifetime_cases']==9 and observed[0]['gc_working_set_epochs']==12 and observed[0]['gc_growth_boundary_reports']==8 and observed[0]['gc_injected_failures_normal_and_sanitized']==48
    print(result.stdout,end='')
print('I verified current c426 Linux284-cache/34-existing-skip/3-worker/6-array, nine lifetime and new normal/sanitized GC controls; complete native/platform/consumer/release acceptance remains separate.')
