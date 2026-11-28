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
assert sha(raw)=='c5203bd50924cb27a4e1bbc0aa6f319a181b137da8181297ff0cb0eacf25ba7f'
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
    results={}
    for platform in ['linux','darwin']:
        expected=json.loads((root/platform/'verified-component.json').read_text())
        result=subprocess.run([sys.executable,str(root/'verify_component.py'),'--root',str(root/platform),'--repo',str(args.repo),'--platform',platform],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        current=json.loads(result.stdout)
        assert current==expected
        assert (current['methods'],current['commands'],current['c_allocation_runs'])==(33,2404,14)
        assert not current['whole_release'] and not current['full_pure_translator']
        results[platform]=current
    assert results['linux']['join_route_report_sha256']==results['darwin']['join_route_report_sha256']
    assert results['linux']['c_allocation_reports']==results['darwin']['c_allocation_reports']
    print('I verify all33 join methods on Linux and Darwin, every actual command/input/tool/stream, original17/36/12 prerequisites and14 C-only allocation observations per host. Full translator and release acceptance remain open.')
