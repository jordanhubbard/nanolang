"""I verify my scoped producer stack-depth component without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='8c6fd7e4eb7bfe313de9bc1cd34329118e9dcecf42f2dea499d9d0e1a0038d49'
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
    for name in ['normal','sanitized']:
        d=load(name+'/result.json')
        assert d['exit']==0 and not d['command_timeout'] and d['guards_match']
        assert d['source_before']==d['source_after'] and d['compiler_before']==d['compiler_after']
        for source in ['src/nanoisa/verifier.c','src/nanoisa/verifier.h','src/nanoisa/nvm_v2_convert.c','tests/nanoisa/test_function_depth_batch.c','tests/nanoisa/test_function_depth_profiles.c']:
            assert sha((root/('source/'+source)).read_bytes())==d['source_before'][source]
        text=(root/(name+'/stdout')).read_text()
        assert 'I passed 17 paired depth cases and output boundaries.' in text
    expanded=load('expanded-controls/result.json')['operations']
    assert [(r['label'],r['exit']) for r in expanded]==[('restore',0),('normal',0),('sanitized',0)]
    for name in ['normal','sanitized']:
        text=(root/('expanded-controls/'+name+'.stdout')).read_text()
        assert 'I passed 9 transient verifier allocation positions and recovery.' in text
        assert 'private-profile depth and fixture controls.' in text
    assert 'undefined reference' in (root/'alloc-expanded.stderr').read_text()
    for name in ['stop-after-refusal','omit-structure']:
        d=load('mutation-'+name+'/result.json')
        assert d['mutation_caught']
        assert [(r['label'],r['exit']) for r in d['operations']]==[('build',0),('run',-6)]
        assert 'Assertion' in (root/('mutation-'+name+'/run.stderr')).read_text()
    old=[json.loads(l) for l in (root/'baseline/depth-count-baseline/run.stdout').read_text().splitlines()]
    new=[json.loads(l) for l in (root/'count-candidate/run.stdout').read_text().splitlines()]
    assert [r['functions'] for r in old]==[r['functions'] for r in new]==[1,8,64,256,1024]
    for a,b in zip(old,new):
        assert a['producer_structure_visits']==a['oracle_structure_visits']==b['oracle_structure_visits']==b['functions']
        assert b['producer_structure_visits']==1 and a['all_depths_match'] and b['all_depths_match']
    assert 'NvmVerifyResult nvm_infer_function_max_stacks' in (root/'source/src/nanoisa/verifier.c').read_text()
    producer=(root/'source/src/nanoisa/nvm_v2_convert.c').read_text()
    assert 'nvm_infer_function_max_stacks(mod, fns, n_fn, sizeof *fns,' in producer
    assert 'test-function-depth-batch' in (root/'source/.github/workflows/c-repair-original-gates.yml').read_text()
print('I verified paired semantics/wire fields, private admission,9 allocator faults, guarded normal/sanitizer passes, both negative mutations and1024-to1 structural visits; platform/release gates remain open.')
