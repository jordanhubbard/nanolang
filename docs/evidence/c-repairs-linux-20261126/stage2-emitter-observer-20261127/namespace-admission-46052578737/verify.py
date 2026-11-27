"""I verify the incomplete observation and local admission repair without a product run."""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent)
parser.add_argument('--workspace', type=Path, required=True)
args = parser.parse_args()
sha = lambda data: hashlib.sha256(data).hexdigest()
raw = (args.archive / 'manifest.json').read_bytes()
assert sha(raw) == '20f4d2bf80dacd3ece70604caa5b502df924af7090c8f854b899795932dd84e2'
manifest = json.loads(raw)
assert not manifest['whole_release'] and not manifest['whole_C_qualification']
assert not manifest['original_timeout_reproduced'] and manifest['recorded_admission_incomplete']
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
    job = json.loads((root / 'job.json').read_text())
    assert (job['id'], job['run_id'], job['head_sha'], job['conclusion']) == (
        140687267273, 46052578737, '9d06f19aebf548a96588d00b6fe5b559da4d0dab', 'success')
    assert sha((root / 'diagnostic.zip').read_bytes()) == (
        '971109e02a452885b647ca661f4411cc01ccd84e4f5bdc04b627fd52222d6529')
    with zipfile.ZipFile(root / 'diagnostic.zip') as archive:
        assert len(archive.namelist()) == 38 and archive.testzip() is None
        prefix = 'stage2-emitter-diagnostic/'
        read = lambda name: archive.read(prefix + name)
        load = lambda name: json.loads(read(name))
        observed = load('diagnostic.json')
        product = observed['product']
        assert observed['observation_valid'] and observed['guards_match']
        assert not observed['qualification'] and not observed['whole_release']
        assert product['terminal'] and product['exit_code'] == 1
        assert not product['command_timeout'] and product['deadline_seconds'] == 300
        assert 0 < product['elapsed_seconds'] < 300
        assert product['output_complete'] and not product['observer_errors']
        assert sha(read('command/stdout')) == product['stdout_sha256']
        assert sha(read('command/stderr')) == product['stderr_sha256']
        assert b'I could not publish complete C-library artifacts for nanoisa' in read('command/stderr')
        assert load('product-temporaries/files.json') == []
        assert not any(name.endswith('/cache-current-restoration.json') for name in archive.namelist())
        before, after = load('identities-before.json'), load('identities-after.json')
        assert before == after
        source_before, source_after = load('observer-sources-before.json'), load('observer-sources-after.json')
        assert source_before == source_after
        for name, expected in source_before.items():
            assert sha(read('controller-sources/' + Path(name).name)) == expected
        old_source = ast.parse(read('observer.py').decode())
        main = next(node for node in old_source.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = [node.func.attr for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == 'helper']
        assert 'restore' in calls and 'restore_cache_currents' not in calls
        assert sha(read('restore-helper.py')) == '5693f86ea998efff0abde49284d362c79c21556cabc6739647438878ce81771f'
        original = read('original-artifact.zip')
        assert sha(original) == 'f80edbf52d706501319cdefa9d5dcdce3ea933b5b3723f2621cf1fffaa258dc5'
        with zipfile.ZipFile(io.BytesIO(original)) as nested:
            assert len(nested.namelist()) == 4066 and nested.testzip() is None
            names = nested.namelist()
            receipt = json.loads(nested.read(next(name for name in names if name.endswith('/bin/nanoc_bootstrap.json'))))
            assert receipt['raw_stage1_stage2_equal'] and len(receipt['steps']) == 17
            assert all(step['exit_code'] == 0 for step in receipt['steps'])
            assert receipt['stage_timeout_seconds'] == 1800 and receipt['shadow_timeout_seconds'] == 30
            hosts = receipt['hosts']
            assert len(hosts) == 6
            for name, expected in hosts.items():
                relative = str(Path(name).relative_to(receipt['root']))
                assert sha(nested.read('nanolang/nanolang/' + relative)) == expected
            assert any('/current/' in name for name in names)
        samples = [json.loads(line) for line in read('command/processes.jsonl').decode().splitlines()]
        assert len(samples) == 111
        assert all(process['group'] == product['pid'] for sample in samples for process in sample['processes'])
    assert re.search(r'Ran 18 tests in [0-9.]+s\n\nOK',
                     (root / 'final-namespace-readiness-controls.stderr').read_text())
    mutation = json.loads((root / 'namespace-omission-mutation.stdout').read_text())
    assert mutation['expected_mutation_failure'] and mutation['tests_run'] == mutation['errors'] == 1
    assert not mutation['product_execution']
    assert '/evidence/flattened/library' in (root / 'namespace-omission-mutation.stderr').read_text()
    corrected = ast.parse((root / 'corrected-observer.py').read_text())
    recovery = next(node for node in corrected.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'restore_admitted_inputs')
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'restore_cache_currents' for node in ast.walk(recovery))
    assert any(isinstance(node, ast.FunctionDef) and node.name == 'native_control' for node in corrected.body)
    print(json.dumps({'verified': True, 'records': len(seen), 'chunks': len(used),
                      'whole_zip_members': 38, 'nested_original_members': 4066,
                      'recorded_byte_guard_entries': len(before), 'input_host_dsos_verified': 6,
                      'recorded_observation_flag': True, 'namespace_admission_complete': False,
                      'product_exit': 1, 'original_300_second_timeout_reproduced': False,
                      'process_samples': len(samples), 'local_protocol_methods_passed': 18,
                      'omission_mutation_caught': True, 'product_execution': False,
                      'whole_C_qualification': False, 'whole_release': False}))
