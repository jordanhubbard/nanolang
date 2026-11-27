"""I verify my CPU sampler controls without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='9ac82f8e9e640867c61e452ef1d58679a4f217ac87d677befec9c593d8a4cb11'
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
    result=json.loads((root/'result.json').read_text())
    assert result['library_before']==result['library_after']
    assert [(r['label'],r['exit']) for r in result['operations']]==[('build',0),('run',0)]
    ids=json.loads((root/'provider-identities.json').read_text())
    assert sha((root/'libnanoisa.so').read_bytes())==ids['library_sha256']==result['library_before']
    assert sha((root/'probe').read_bytes())==ids['probe_sha256']
    rows=[json.loads(line) for line in (root/'run.stdout').read_text().splitlines()]
    assert [r['functions'] for r in rows]==[1,8,64,256,1024]
    assert all(r['oracle_structure_visits']==r['producer_structure_visits']==r['functions'] and r['all_depths_match'] for r in rows)
    old=(root/'original-verifier.c').read_text();instrumented=(root/'verifier.c').read_text()
    needle='static NvmVerifyResult verify_module_ranges(const NvmModule *mod) {'
    assert old.count(needle)==1
    assert old.replace(needle,'size_t nano_probe_range_calls;\n'+needle+'\n    ++nano_probe_range_calls;')==instrumented
    probe=(root/'probe.c').read_text()
    assert 'depth==i%7+1' in probe and 'max_stack==i%7+1' in probe
    assert 'nvm_private_from_module(module,&converted,NULL)' in probe
    assert 'nvm_verify_function_max_stack(mod, i, &depth).ok' in (root/'original-producer.c').read_text()
print('I verified the actual ordinary producer repeats whole-module structure1/8/64/256/1024 times while preserving every independently known depth; no optimization or platform qualification is claimed.')
