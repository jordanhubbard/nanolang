"""I verify current successful Linux and failed Darwin bootstrap cohorts without executing products."""
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
assert sha(raw)=='f600befd87932a33f409380537579220ffad1d542e349168c6c5b08910341157'
manifest=json.loads(raw)
assert not manifest['qualification_execution'] and not manifest['whole_release']
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
    for report,reader in [('verified-linux-bootstrap.json','verify_linux_bootstrap.py'),('verified-darwin-bootstrap-failure.json','verify_darwin_bootstrap_failure.py')]:
        observed=json.loads((root/report).read_text())
        result=subprocess.run([sys.executable,str(root/reader),'--root',str(root),'--repo',str(args.repo)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert observed==json.loads((root/report).read_text())
        print(result.stdout,end='')
    linux=json.loads((root/'verified-linux-bootstrap.json').read_text())[0]
    darwin=json.loads((root/'verified-darwin-bootstrap-failure.json').read_text())[0]
    assert linux['full_original_native_gate_passed'] and linux['line_splitting_methods']==2
    assert not darwin['job_success'] and not darwin['full_original_native_gate_passed'] and darwin['emitter_methods']==0
    assert not darwin['diagnostic_qualification'] and darwin['diagnostic_completed_shadows']==1139
print('I verified current Linux17/36/12/five-route success and retained Darwin17/36 plus ordinary C-seed shadow10 refusal. A1139-shadow diagnostic pass grants no Darwin qualification or original failure cause; full release remains open.')
