"""I verify my current Darwin bootstrap cohort without executing a product."""
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
assert sha(raw)=='20c461d899d5e83b2a097c243f470d41e32917d0f56478a2c38e670e834eb1c2'
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
    observed=json.loads((root/'verified-darwin-bootstrap.json').read_text())
    result=subprocess.run([sys.executable,str(root/'verify_darwin_bootstrap.py'),'--root',str(root),'--repo',str(args.repo)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert observed==json.loads((root/'verified-darwin-bootstrap.json').read_text())
    assert len(observed)==1
    q=observed[0];assert q['bootstrap_steps']==17 and q['raw_module_equal'] and q['installed_image_matches_stage2'] and q['no_c_seed_probe']
    assert q['native_methods']==36 and q['emitter_methods']==12 and q['line_splitting_methods']==2 and q['host_libraries']==6 and q['git_source_files']==1382
    assert q['native_macho_arm64'] and q['gc_working_set_epochs']==12 and q['gc_growth_boundary_reports']==8 and q['gc_injected_failures_normal_and_sanitized']==48
    print(result.stdout,end='')
print('I verified original c426 Darwin ARM64 bootstrap17/raw equality/installed/no-C-seed/36/12, two five-route methods and mandatory GC/lifetime controls. Other current gates and full release remain separate.')
