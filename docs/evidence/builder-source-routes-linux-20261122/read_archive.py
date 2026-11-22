#!/usr/bin/env python3
"""I verify my source checkpoint without promoting its open acceptance gates."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import os
import subprocess

here = Path(__file__).resolve().parent
root = here.parents[2]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
m = json.loads((here/'manifest.json').read_text())
rows = {r['path']: r for r in m['files']}
assert len(rows) == m['counts']['files']
objects = {r['artifact']: r for r in rows.values()}
assert len(objects) == m['counts']['objects']
assert sum(r['bytes'] for r in rows.values()) == m['counts']['raw_bytes']
assert sum(r['compressed_bytes'] for r in objects.values()) == m['counts']['compressed_bytes']
verified = {}
by_digest = {}
for name, row in rows.items():
    by_digest.setdefault(row['sha256'], row)
for base in m['base_archives']:
    path = here/base['manifest']
    assert sha(path.read_bytes()) == base['sha256']
    data = json.loads(path.read_text())
    for item in data['files']:
        row = dict(item, artifact=os.path.relpath(path.parent/item['artifact'], here))
        by_digest.setdefault(row['sha256'], row)

def read_row(row):
    artifact = row['artifact']
    identity = tuple(row[k] for k in ['bytes', 'sha256', 'compressed_bytes', 'compressed_sha256'])
    packed = (here/artifact).read_bytes()
    assert len(packed) == row['compressed_bytes'] and sha(packed) == row['compressed_sha256'], artifact
    raw = gzip.decompress(packed)
    assert len(raw) == row['bytes'] and sha(raw) == row['sha256'], artifact
    if artifact in verified:
        assert verified[artifact] == identity, artifact
    verified[artifact] = identity
    return raw

def read(name):
    return read_row(rows[name])

def data(name):
    return json.loads(read(name))

for name, row in rows.items():
    identity = tuple(row[k] for k in ['bytes', 'sha256', 'compressed_bytes', 'compressed_sha256'])
    if row['artifact'] in verified:
        assert verified[row['artifact']] == identity, name
    else:
        read_row(row)

def available(digest):
    assert digest in by_digest, digest
    row = by_digest[digest]
    if row['artifact'] not in verified:
        read_row(row)
    return row

# I compare every declared current source to the exact published source commit.
source_names = sorted(name.split('/', 1)[1] for name in rows if name.startswith('source/'))
entries = subprocess.check_output(['git', 'ls-tree', '-rz', m['source_commit'], '--', *source_names], cwd=root)
tree = {}
for entry in entries.split(b'\0'):
    if entry:
        meta, path = entry.split(b'\t', 1)
        tree[path.decode()] = meta.split()[2].decode()
for path in source_names:
    raw = read('source/'+path)
    if path in tree:
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == tree[path], path
    else:
        assert path == 'modules/index.json', path
assert subprocess.run(['git', 'merge-base', '--is-ancestor', m['source_base_commit'], m['source_commit']], cwd=root).returncode == 0

def check_guard(before_name, after_name, status=None, current=False):
    before, after = data(before_name), data(after_name)
    if status is not None:
        assert after.pop('status') == status, after_name
    assert before == after, (before_name, after_name)
    for name, digest in before['sources'].items():
        available(digest)
        if current:
            assert rows['source/'+name]['sha256'] == digest, name
    for kind in ['tools', 'root_tools', 'compiler', 'compilers']:
        for name, digest in before.get(kind, {}).items():
            available(digest)
    return before

# I retain failures as failures, including the independent callable audit.
old = 'observation/integration-neighbors/'
check_guard(old+'neighbors-before.json', old+'original-build-units-after.json')
r = data(old+'neighbor-cohorts.json')
assert len(r) == 1 and r[0]['returncode'] == 2
old_log = read(old+'original-build-units.log')
assert b'Building typecheck...' in old_log and b'I stopped shadow execution after 10 seconds.' in old_log
assert b'Stage 2: 3/3 components built successfully' not in old_log

new = 'observation/indexed-original-build/'
build_guard = check_guard(new+'before.json', new+'original-build-units-after.json', current=True)
ops = data(new+'operations.json')
assert [r['returncode'] for r in ops] == [0, 2]
log = read(new+'original-build-units.log')
assert sha(log) == ops[1]['log_sha256']
assert b'Stage 2: 3/3 components built successfully' in log
assert b'Stage 3: 3/3 component entry checks passed' in log
assert b'All eval tests passed!' in log
assert b'I retain 576 callable names across nine 64-call routes.' in log
assert b'test-eval-callable-lifetimes' in log
remaining = 'observation/indexed-remaining-units/'
check_guard(remaining+'before.json', remaining+'after.json',
            'PASS_ORIGINAL_REMAINING_ENV_UNITS_PURE_SEED_REGENERATION_STILL_REQUIRED', current=True)
assert data(remaining+'operations.json')[0]['returncode'] == 0
assert b'60 passed, 0 failed' in read(remaining+'original-env-scope-and-index.log')

# I keep the existing-ten-second observation separate from original acceptance.
diag = 'observation/component-observation-1200/'
check_guard(diag+'before.json', diag+'after.json', 'STABLE_DIAGNOSTIC_ONLY')
observed = data(diag+'result.json')
assert observed['returncode'] == 1 and observed['prior_sha256'] == observed['output_sha256']
assert observed['qualification'] == 'DIAGNOSTIC_ONLY_ORIGINAL_FULL_BUILD_FAILURE_RETAINED'
assert not data(diag+'shadows.json')['completed']
events = []
for line in read(diag+'stderr').decode().splitlines():
    if line.startswith('{'):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
completed = [r for r in events if r.get('event') == 'shadow-complete']
assert len(completed) == 504 and all(r['failures'] == 0 for r in completed)
assert events[-1]['event'] == 'shadow-start' and events[-1]['name'] == 'resource_callback_unsupported'
assert data(diag+'samples.json')['samples']
for name in ['observation/import-lookup-linear-fixed.log', new+'import-lookup-indexed.log']:
    samples = [json.loads(line) for line in read(name).decode().splitlines() if line.startswith('{')]
    assert [r['imports'] for r in samples] == [0, 4, 64, 1024]
    assert all(r['queries'] == 100000 and r['cpu_seconds'] > 0 for r in samples)
assert b"undefined reference to `g_argc'" in read('observation/import-lookup-linear-link-failure.log')

# I require the complete current cohorts, actual method counts and unchanged guards.
cohort_counts = []
for folder, specs in [
    ('neighbors-indexed', [('public-gcc', 9), ('map-gcc', 8), ('introspection-c', 13), ('introspection-pure', 13), ('public-clang', 6), ('compiler-contracts', None)]),
    ('additional-indexed', [('header-callbacks', 8), ('map-contexts-current-producers', 7)])]:
    prefix = 'observation/'+folder+'/'
    before = check_guard(prefix+'neighbors-before.json', prefix+'neighbors-after.json', 'PASS', current=True)
    cohorts = data(prefix+'neighbor-cohorts.json')
    assert [(r['label'], r['expected_methods']) for r in cohorts] == specs
    for row in cohorts:
        assert row['returncode'] == 0
        body = read(prefix+row['label']+'.log')
        assert sha(body) == row['log_sha256']
        assert data(prefix+row['label']+'-after.json') == before
        if row['expected_methods'] is not None:
            result = data(prefix+'observer-'+row['label']+'/result.json')
            assert result['discovered'] == result['ran'] == row['expected_methods']
            assert result['successful'] and not result['failures'] and not result['errors'] and not result['skipped']
            assert body.rstrip().endswith(b'OK')
            cohort_counts.append(row['expected_methods'])

source = 'observation/source-indexed/'
final = check_guard(source+'source-before.json', source+'source-after.json',
                    'PASS_THREE_PRODUCER_ROUTES_NATIVE_ABI_PENDING_HANDOFF', current=True)
source_ops = data(source+'source-operations.json')
assert len(source_ops) == 57 and all(r['returncode'] == 0 for r in source_ops)
assert sum(r['label'].endswith('-legacy-run') for r in source_ops) == 7
assert sum(r['label'].endswith('-c-vm') for r in source_ops) == 7
assert sum(r['label'].endswith('-pure-vm') for r in source_ops) == 7
assert source_ops[-1]['label'] == 'descriptor-kind-controls'
assert b'Ran 3 tests in ' in read(source+'descriptor-kind-controls.stderr')

seed = data('observation/seed-indexed/manifest.json')
assert seed['status'] == 'passed'
b = seed['bootstrap_manifest']
assert b['stage_timeout_seconds'] == 1800 and b['shadow_timeout_seconds'] == 30
assert [r['label'] for r in b['steps']] == ['seed', 'seed-verify']
assert len(b['hosts']) == 6
for step in b['steps']:
    assert step['exit_code'] == 0
    assert sha(read('bootstrap-seed-indexed/'+Path(step['log']).name)) == step['log_sha256']
for name, digest in b['sources'].items():
    assert rows['source/'+name]['sha256'] == digest
for name, digest in {**b['tools'], **b['artifacts']}.items():
    assert final['tools'][name] == digest
    available(digest)
for name, digest in b['hosts'].items():
    available(digest)
assert not any(name.startswith('bootstrap-seed-indexed/') and name.rsplit('/', 1)[-1] in
               ['stage1.nvm', 'stage2.nvm', 'unexpected-native-compiler'] for name in rows)

closed = 'observation/closed-builder-consumer-indexed/'
check_guard(closed+'before.json', closed+'after.json',
            'CLOSED_BUILDER_ADAPTER_BOUNDARY_PRIOR_OUTPUT_PRESERVED', current=True)
closed_ops = data(closed+'operations.json')
assert [r['producer'] for r in closed_ops] == ['c', 'pure']
for row in closed_ops:
    assert row['returncode'] > 0 and row['prior_sha256'] == row['after_sha256']
    assert b'typed value adapter' in read(closed+row['producer']+'.stderr')

private = 'observation/private-module-bindings-layout/'
g = check_guard(private+'before.json', private+'after.json',
                'PASS_FOCUSED_COMPILER_ASAN_UBSAN_NOT_LEAK_QUALIFICATION', current=True)
available(g['private_compiler'])
for digest in g['instrumented_objects'].values():
    available(digest)
ops = data(private+'operations.json')
assert [(r['label'], r['phase'], r['returncode']) for r in ops] == [
    ('callback', 'compile', 0), ('callback', 'run', 0), ('graph', 'compile', 0), ('graph', 'run', 0)]
for label in ['callback', 'graph']:
    report = data(private+label+'-shadow.json')
    assert report['success'] and report['completed']
assert b'runtime/nl_string.h: No such file or directory' in read('observation/private-module-bindings/callback-compile.stderr')

# I never substitute a hash-only historical reference for actual retained bytes.
for row in m['unavailable_historical_references']:
    assert row['sha256'] not in by_digest, row
assert 'src/nanoisa/nvm2c.c' in source_names
print(json.dumps(dict(status='INTEGRITY_PASS', qualification='PARTIAL_OPEN_C_ADAPTER_1140_1198_RAW_STAGES_PAIRED_PLATFORMS',
    source_commit=m['source_commit'], records=len(rows), raw_bytes=m['counts']['raw_bytes'],
    current_python_methods=sum(cohort_counts)+3, builder_positive_programs=7,
    builder_producer_routes_per_program=3, strict_callable_names_retained=576,
    unavailable_historical_references=len(m['unavailable_historical_references'])), indent=2))
