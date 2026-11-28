"""I verify the rejected-declaration repair without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='e5dd3aa01cea0fd98153e7aa086b86baada9972014168bf81b8bf3d639274c1a'
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
    gdb=load('baseline-gdb.json')
    assert gdb['compiler_before']==gdb['compiler_after']=='f2315716d9b292d9c748a17d628a21e0006b39efef7b1b881f3ea0406da6ed86'
    assert 'Program received signal SIGSEGV' in (root/'baseline-gdb.stdout').read_text()
    assert 'type_check' in (root/'baseline-gdb.stdout').read_text()
    baseline=load('baseline/result.json')['operations']
    assert [(row['label'],row['exit']) for row in baseline]==[('link',0),('unit',-11),('cli',1)]
    assert 'FAILED (failures=4)' in (root/'baseline/cli.stderr').read_text()
    expected={('legal-name',False,True),('legal-name',True,True),('orphan-shadow',False,False),('orphan-shadow',True,False),('reserved-with-shadow',False,False),('reserved-without-shadow',False,False)}
    for name in ['normal','sanitized']:
        d=load(name+'/result.json')
        assert d['exit']==0 and not d['command_timeout'] and d['guards_match']
        assert d['source_before']==d['source_after'] and d['compiler_before']==d['compiler_after']
        rows=[json.loads(line) for line in (root/(name+'/stdout')).read_text().splitlines() if line.startswith('{"case"') and '"accepted"' in line]
        assert len(rows)==6 and {(r['case'],r['module'],r['accepted']) for r in rows}==expected
    normal=(root/'normal/stderr').read_text()
    assert 'Ran 2 tests in ' in normal and 'FAILED' not in normal
    mutation=load('warning-guard-mutation/result.json')
    assert mutation['mutation_caught'] and mutation['candidate_source_unchanged']
    assert [(row['label'],row['exit']) for row in mutation['operations']]==[('compile',0),('link',0),('run',1)]
    assert 'member access within null pointer' in (root/'warning-guard-mutation/run.stderr').read_text()
    old=(root/'source/baseline-typechecker.c').read_text();new=(root/'source/src/typechecker.c').read_text()
    assert new.replace('            /* I do not warn through a declaration that registration rejected. */\n','').replace('if (func && !env->suppress_shadow_warnings && !env->gpu_target &&','if (!env->suppress_shadow_warnings && !env->gpu_target &&',1)==old
    ids=load('provider-identities.json')
    assert sha(old.encode())==ids['baseline_typechecker_sha256'] and sha(new.encode())==ids['candidate_typechecker_sha256']
    test=(root/'source/tests/test_checker_rejected_declarations.py').read_text()
    assert 'self.assertGreater(result.returncode, 0' in test and "self.assertEqual(output.read_bytes(), b'prior-native-output')" in test
    assert 'test-checker-rejected-declarations test-checker-rejected-declaration-cli' in (root/'source/.github/workflows/c-repair-original-gates.yml').read_text()
print('I verified the rejected-declaration fix, original crashes, six checker cases, CLI publication controls, sanitizer checks and causal mutation; platform/release qualification remains open.')
