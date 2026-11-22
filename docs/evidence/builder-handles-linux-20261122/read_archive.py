#!/usr/bin/env python3
"""I check my Builder foundation without converting a missing GCC gate into PASS."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess

here = Path(__file__).resolve().parent
root = here.parents[2]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
m = json.loads((here/'manifest.json').read_text())
assert sha((here/m['base_manifest']).read_bytes()) == m['base_manifest_sha256']
rows = {r['path']: r for r in m['files']}
assert len(rows) == m['counts']['files']
objects = {r['artifact'] for r in rows.values()}
assert len(objects) == m['counts']['objects']
assert sum((here/p).stat().st_size for p in objects) == m['counts']['compressed_bytes']

def read(name):
    r = rows[name]
    packed = (here/r['artifact']).read_bytes()
    assert len(packed) == r['compressed_bytes'] and sha(packed) == r['compressed_sha256'], name
    raw = gzip.decompress(packed)
    assert len(raw) == r['bytes'] and sha(raw) == r['sha256'], name
    return raw

def data(name):
    return json.loads(read(name))

# Repeated logical paths may reference one immutable compressed object. I check
# its bytes once and require every referring record to declare the same identity.
verified_objects = {}
for name, row in rows.items():
    identity = tuple(row[key] for key in ['bytes', 'sha256', 'compressed_bytes', 'compressed_sha256'])
    if row['artifact'] in verified_objects:
        assert verified_objects[row['artifact']] == identity, name
    else:
        read(name)
        verified_objects[row['artifact']] = identity
assert sum(r['bytes'] for r in rows.values()) == m['counts']['raw_bytes']

def git_tree(commit):
    result = {}
    for entry in subprocess.check_output(['git', 'ls-tree', '-rz', commit], cwd=root).split(b'\0'):
        if entry:
            meta, path = entry.split(b'\t', 1)
            result[path.decode()] = meta.split()[2].decode()
    return result

trees = {phase: git_tree(m[phase]) for phase in ['source_commit', 'foundation_commit']}
for name in rows:
    if name.startswith(('source/', 'foundation-source/')):
        prefix, path = name.split('/', 1)
        tree = trees['foundation_commit' if prefix == 'foundation-source' else 'source_commit']
        raw = read(name)
        if path in tree:
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == tree[path], name
        else:
            assert path == 'modules/index.json', name

def source(phase, path):
    return read(('foundation-source/' if phase == 'foundation' and path == 'Makefile.gnu' else 'source/')+path)

def check_guard(phase, stem, status):
    before = data('observation/'+stem+'-before.json')
    after = data('observation/'+stem+'-after.json')
    assert after.pop('status') == status and before == after
    for name, digest in before['sources'].items():
        assert sha(source(phase, name)) == digest, name
    for name, digest in before['tools'].items():
        assert sha(read('tools/'+digest+'/'+name)) == digest
    for name, digest in before['compilers'].items():
        assert sha(read('compilers/'+digest+'/'+Path(name).name)) == digest
    return before

foundation = check_guard('foundation', 'qualification', 'STABLE_WITH_OPEN_GCC_TSAN_BOUNDARY')
integration = check_guard('integration', 'make', 'PASS')
for stem, guard, labels, counts in [
    ('cohorts', foundation, ['gcc-builder', 'clang-native-source', 'gcc-source-abi', 'neighbors'], [1, 12, 14, None]),
    ('make-cohorts', integration, ['make-clang', 'make-source-gcc', 'make-source-clang'], [2, 8, 8])]:
    cohorts = data('observation/'+stem+'.json')
    assert [r['label'] for r in cohorts] == labels
    assert [r['expected_methods'] for r in cohorts] == counts
    for r in cohorts:
        raw = read('observation/'+r['label']+'.log')
        assert sha(raw) == r['log_sha256']
        assert data('observation/'+r['label']+'-after.json') == guard
        if r['label'] == 'gcc-builder':
            assert r['returncode'] == 1 and r['followup'] == 1198
            assert r['status'] == 'RETAINED_GCC_TSAN_MAPPING_REFUSAL'
            case = 'native/'+Path(r['case']).name
            ops = data(case+'/operations.json')
            assert ops[-1]['label'] == 'run-threaded' and ops[-1]['returncode'] == 66
            assert b'ThreadSanitizer: unexpected memory mapping' in read(case+'/run-threaded.stderr')
            for op in ops[:-1]:
                assert op['returncode'] == (1 if op['fault_child'] else 0)
        else:
            assert r['returncode'] == 0
            if r['expected_methods'] is not None:
                assert ('Ran '+str(r['expected_methods'])+' tests in ').encode() in raw
                assert raw.rstrip().endswith(b'OK')

# I check both guarded seeds at their own source/Make pins; neither is a raw stage receipt.
for version, phase, guard in [('seed-current', 'foundation', foundation), ('seed-make', 'integration', integration)]:
    seed = data('observation/'+version+'/manifest.json')
    assert seed['status'] == 'passed'
    b = seed['bootstrap_manifest']
    assert b['stage_timeout_seconds'] == 1800 and b['shadow_timeout_seconds'] == 30
    assert [r['label'] for r in b['steps']] == ['seed', 'seed-verify']
    for r in b['steps']:
        assert r['exit_code'] == 0
        assert sha(read('bootstrap-'+version+'/'+Path(r['log']).name)) == r['log_sha256']
    for name, digest in b['sources'].items():
        assert sha(source(phase, name)) == digest
    assert len(b['hosts']) == 6
    for name, digest in b['hosts'].items():
        assert sha(read('hosts/'+digest+'/'+Path(name).name)) == digest
    assert b['artifacts']['nanoc_seed.nvm'] == guard['tools']['nanoc_seed.nvm']
    for name, digest in b['tools'].items():
        assert guard['tools'][name] == digest

baseline = data('observation/baseline-operations.json')
assert [r['returncode'] for r in baseline] == [0, 0, 1, 1]
assert b'heap-use-after-free' in read('observation/stale.stderr')
assert b'misaligned address' in read('observation/wrong-kind.stderr')
assert data('observation/baseline-guards.json')['source_commit'] == '46df27b0ac1ba95c2591bf2d6cb7efc29b410f4e'
bg = data('observation/baseline-guards.json')
assert sha(read('observation/baseline')) == bg['binary_sha256']
baseline_tree = git_tree(bg['source_commit'])
for path, digest in bg['inputs'].items():
    raw = read('observation/baseline-source/'+path)
    assert sha(raw) == digest
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == baseline_tree[path]
entry = data('observation/tsan-entry-operations.json')
assert [(r['returncode'], r['main_marker']) for r in entry] == [(0, False), (66, False), (0, False), (0, True)]
assert b'ThreadSanitizer: unexpected memory mapping' in read('observation/gcc-entry-run.stderr')
assert b'ThreadSanitizer: unexpected memory mapping' in read('observation/gcc-original-handles.log')

# I keep actual image custody for failed and passing native products.
completed_builders = []
for name in rows:
    if name.startswith('native/builder-handles-') and name.endswith('/custody-after.json'):
        c = data(name)
        assert c['inputs_before'] == c['inputs_after'] and c['compiler_before'] == c['compiler_after']
        for path, digest in c['inputs_before'].items():
            assert sha(read('source/'+path)) == digest
        assert c['compiler_before'] in foundation['compilers'].values()
        for binary, identity in c['products'].items():
            assert identity['sha256_before'] == identity['sha256_after']
            assert sha(read(name.rsplit('/', 1)[0]+'/'+binary)) == identity['sha256_after']
        if name.rsplit('/', 1)[0]+'/products.json' in rows:
            completed_builders.append(name)
assert len(completed_builders) >= 2
for name in completed_builders:
    parent = name.rsplit('/', 1)[0]
    for op in data(parent+'/operations.json'):
        assert op['returncode'] == (1 if op['fault_child'] else 0)
first = data('observation/first-fixtures/identity.json')
for path, digest in first['inputs'].items():
    assert sha(read('observation/first-fixtures/'+path)) == digest
initial_guard = data('native/builder-handles-jmbz519b/inputs-before-after.json')
for path, digest in first['inputs'].items():
    assert initial_guard[path] == digest
initial_parent = 'native/builder-handles-jmbz519b'
for name, identity in data(initial_parent+'/products.json').items():
    assert set(identity) == {'sha256_before_after'}
    assert sha(read(initial_parent+'/'+name)) == identity['sha256_before_after']
for op in data(initial_parent+'/operations.json'):
    assert op['returncode'] == (1 if op['fault_child'] else 0)
old_make = read('foundation-source/Makefile.gnu')
new_make = read('source/Makefile.gnu')
assert old_make.replace(b'python3 -m unittest tests.test_collection_handles\n', b'python3 -m unittest tests.test_collection_handles tests.test_builder_handles\n') == new_make
print(json.dumps(dict(status='PASS', qualification='PARTIAL_WITH_OPEN_GCC_TSAN_GATE', source_commit=m['source_commit'], foundation_commit=m['foundation_commit'], records=len(rows), raw_bytes=m['counts']['raw_bytes'], clang_methods=12, gcc_source_abi_methods=14, make_methods=2, integrated_source_methods_per_compiler=8), indent=2))
