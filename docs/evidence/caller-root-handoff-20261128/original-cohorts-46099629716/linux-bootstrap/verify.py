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
assert sha(raw)=='4826d3beae8b8b8c30f6325719f677935f5caac466b7355a1ac35ef73b1718c8'
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
    label='linux-bootstrap'
    observed=json.loads((root/('verified-'+label+'.json')).read_text())
    result=subprocess.run([sys.executable,str(root/'verify_linux_bootstrap.py'),'--root',str(root),'--repo',str(args.repo)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert observed==json.loads((root/('verified-'+label+'.json')).read_text()) and len(observed)==1
    q=observed[0];assert q['source_tool_image_guards_verified'] and q['caller_root_controls_normal'] and not q['whole_release']
    assert q['gc_working_set_epochs']==12 and q['gc_injected_failures_normal_and_sanitized']==48
    if label=='linux-bootstrap':
        assert not q['job_success'] and q['bootstrap_steps']==17 and q['native_methods']==36 and q['emitter_methods']==0
        assert not q['full_original_native_gate_passed'] and q['emitter_setup_timeout_seconds']==300
    elif label=='darwin-bootstrap':
        assert q['job_success'] and q['bootstrap_steps']==17 and q['native_methods']==36 and q['emitter_methods']==12
        assert q['full_original_native_gate_passed'] and q['line_splitting_methods']==2
    else:
        assert q['job_success'] and q['cache_reported_methods']==284 and q['worker_methods']==3 and q['array_methods']==6
        assert q['full_original_cache_worker_gate_passed']
    print(result.stdout,end='')
print('I verified the complete retained source54 cohort, exact source/tool/image guards and mandatory ordinary caller-root controls. I preserve its actual terminal outcome and grant no complete release acceptance.')
