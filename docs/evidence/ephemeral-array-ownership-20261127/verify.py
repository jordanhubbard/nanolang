"""I verify my temporary-array red baseline without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='56161c9c3de3bb111ebb50c97b8a8fe0d28df157f1681f6f1584d3145f3ef2b2'
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
    expected=[('scalar-argument',2,1),('string-argument',2,1),('nested-argument',5,2),('empty-local-argument',2,1),('repeated-arguments',65,1),('early-return',2,1),('returned-local-alias',2,1),('global-escape-and-replacement',2,1),('shadow-only-argument',1,0),('global-shared-positive',1,1)]
    for prefix in ['', 'sanitized/']:
        d=load(prefix+'result.json');assert not d['qualification'] and not d['whole_release']
        assert [(r['label'],r['exit']) for r in d['operations']]==[('build',0),('run',1)]
        assert d['inputs_before']==d['inputs_after']
        assert d['test_sha256']==sha((root/'source/tests/test_eval_ephemeral_arrays.c').read_bytes())
        log=(root/(prefix+'run.stdout')).read_text()
        rows=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
        assert [(r['case'],r['expected_reclaims'],r['actual_reclaims']) for r in rows]==expected
        assert sum(r['expected_reclaims']-r['actual_reclaims'] for r in rows)==74
        assert 'I preserve caller-supplied and C-caller-returned array storage.' in log
        assert 'I found 9 temporary-array ownership failures.' in log
    d=load('result.json');assert d['source_before']==d['source_after']
    match=load('plain-provider-source-match.json');assert match['source_match'] and match['sources']==d['source_before']
    d=load('sanitized/result.json');assert d['ASAN_OPTIONS']=='detect_leaks=1:halt_on_error=1' and d['UBSAN_OPTIONS']=='halt_on_error=1'
    provider=load('provider/sanitized-result.json');assert provider['exit']==0 and provider['guards_match']
    assert d['source_provider_receipt_sha256']==sha((root/'provider/sanitized-result.json').read_bytes())
    for name,digest in d['direct_object_sources'].items():assert provider['source_before'][name]==digest
    assert set(d['source_provider_differences'])<=set(['src/nanoisa/verifier.c','src/nanoisa/verifier.h','src/nanoisa/nvm_v2_convert.c'])
    log=(root/'sanitized/run.stderr').read_text()
    assert 'LeakSanitizer: detected memory leaks' in log
    assert 'SUMMARY: AddressSanitizer: 3024 byte(s) leaked in 149 allocation(s).' in log
    assert 'ERROR: AddressSanitizer:' not in log and 'runtime error:' not in log
print('I verified nine red ownership cases,74 missing owners, passing caller/alias controls and leak-enabled3024-byte/149-allocation evidence. No lifetime repair or release acceptance follows.')
