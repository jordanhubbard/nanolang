#!/usr/bin/env python3
"""I check my bounded collector receipt without admitting raw or native stages."""
from pathlib import Path
import ast
import gzip
import hashlib
import io
import json
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    files = {}
    artifacts = set()
    for row in manifest['files']:
        assert row['path'] not in files
        encoded = (HERE / row['artifact']).read_bytes()
        assert len(encoded) == row['compressed_bytes']
        assert sha(encoded) == row['compressed_sha256']
        assert row['encoding'] in ('gzip', 'identity')
        data = gzip.decompress(encoded) if row['encoding'] == 'gzip' else encoded
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
        files[row['path']] = data
        artifacts.add(row['artifact'])
    assert artifacts == {str(p.relative_to(HERE)) for p in (HERE / 'objects').iterdir()}

    def read(name):
        return files['observation/' + name]

    def document(name):
        return json.loads(read(name))

    after = document('after.json')
    before = document('seed-before.json')
    assert after['status'] == 'PASS_SEED_VERIFY_AND_STABLE_GUARDS_CROSS_RUN_IMPORT_PATH_DIFFERENCE_RETAINED'
    assert not after['raw_seed_byte_identical']
    assert {k: v for k, v in after['guards'].items() if k != 'seed'} == {
        k: v for k, v in before.items() if k != 'seed'}
    comparison = document('source-comparison.json')
    assert comparison['same_selected_paths'] and comparison['selected'] == 1342
    assert set(comparison['differences']) == {'scripts/bootstrap_nanoisa.py'}
    assert comparison['original_boundary_assertions_unchanged']
    for version in ('312', '315'):
        assert b'FAILED (failures=2)' in read('original-controls-python' + version + '.log')
        assert b'\nOK\n' in read('controls-python' + version + '.log')
    dependency = document('bootstrap-dependencies.json')
    assert dependency['returncode'] == 0
    log = read('bootstrap-dependencies.log')
    assert sha(log) == dependency['log_sha256']
    assert b'Ran 5 tests' in log and b'Ran 27 tests' in log
    assert log.count(b'\nOK\n') == 2 and b'skipped=' not in log
    rebuild = document('tool-rebuild-boundary.json')
    assert set(rebuild['changed_tools']) == {'nano_vm', 'nvm2c', 'nano_aot_runtime.o'}
    assert rebuild['gate_returncode'] == 0 and rebuild['qualification_wrapper_returncode'] == 1
    assert b'AssertionError' in read('qualification.log')
    assert b'AssertionError' in read('seed-qualification.log')
    for name, expected in before['tools'].items():
        assert sha(gzip.decompress(read('rebuilt-tools/' + name + '.gz'))) == expected
    seed = after['current_seed_manifest']
    assert seed['stage_timeout_seconds'] == 1800 and seed['shadow_timeout_seconds'] == 30
    assert [(step['label'], step['exit_code']) for step in seed['steps']] == [
        ('seed', 0), ('seed-verify', 0)]
    assert seed['sources'] == before['sources']
    assert seed['tools'] == {name: before['tools'][name] for name in seed['tools']}
    assert sha(read('previous-seed.nvm')) == before['seed']
    assert sha(read('current-seed.nvm')) == after['current_seed'] == seed['artifacts']['nanoc_seed.nvm']
    old_hosts = {Path(path).name: (path, value) for path, value in before['hosts'].items()}
    new_hosts = {Path(path).name: (path, value) for path, value in seed['hosts'].items()}
    assert set(old_hosts) == set(new_hosts) and len(new_hosts) == 6
    dumps = []
    for label, hosts in (('previous', old_hosts), ('current', new_hosts)):
        dump = read(label + '-seed.dump')
        for name, (path, value) in hosts.items():
            assert old_hosts[name][1] == new_hosts[name][1]
            assert sha(files['current-hosts/' + name]) == value
            dump = dump.replace(path.encode(), ('<retained-host:' + name + '>').encode())
        dumps.append(dump)
    assert dumps[0] == dumps[1]
    vm = document('nanovm.json')
    assert vm['returncode'] == 0 and vm['guards_unchanged']
    vm_log = read('nanovm.log')
    assert sha(vm_log) == vm['log_sha256']
    assert b'569591 passed, 0 failed, 569591 total' in vm_log
    assert b'I passed checked heap allocation failure and recovery checks.' in vm_log
    assert b'I passed stack reserve overflow, failure, entry atomicity and recovery checks.' in vm_log

    expected = dict(before['sources'])
    expected.update({name.removeprefix('source-tests/'): sha(data)
                     for name, data in files.items() if name.startswith('source-tests/')})
    for name, value in before['sources'].items():
        assert sha(files['selected-sources/' + name]) == value
    tracked = set(subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', manifest['source_commit']], cwd=ROOT, text=True).splitlines())
    generated = set(expected) - tracked
    assert not any(name.startswith('tests/') for name in generated)
    archive = subprocess.check_output(['git', 'archive', manifest['source_commit'], '--',
                                       *sorted(set(expected) & tracked)], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        actual = {entry.name: sha(source.extractfile(entry).read())
                  for entry in source if entry.isfile()}
    assert actual == {name: value for name, value in expected.items() if name in tracked}
    original = read('bootstrap_nanoisa-original.py').decode()
    current = subprocess.check_output(['git', 'show', manifest['source_commit'] +
                                       ':scripts/bootstrap_nanoisa.py'], cwd=ROOT, text=True)

    def unchanged_collector_neighbors(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body
                     if not isinstance(node, ast.FunctionDef) or node.name != 'source_inputs']
        return ast.dump(tree, include_attributes=False)

    assert unchanged_collector_neighbors(original) == unchanged_collector_neighbors(current)
    old_tests = subprocess.check_output(['git', 'show', manifest['previous_checkpoint'] +
                                         ':tests/test_bootstrap_nanoisa.py'], cwd=ROOT, text=True)
    new_tests = files['source-tests/tests/test_bootstrap_nanoisa.py'].decode()

    def boundaries(source):
        return ast.dump(next(node for node in ast.parse(source).body
                             if isinstance(node, ast.ClassDef) and node.name == 'BootstrapBoundaries'),
                        include_attributes=False)

    assert boundaries(old_tests) == boundaries(new_tests)
    result = dict(status='INTEGRITY_PASS', source_commit=manifest['source_commit'],
                  records=len(files), source_paths=len(before['sources']),
                  selected_tracked_sources=len(set(before['sources']) & tracked),
                  selected_generated_sources=len(generated),
                  original_make_methods=32, vm_assertions=569591,
                  raw_seed_byte_identical=False,
                  qualification=manifest['qualification'],
                  release_acceptance=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
