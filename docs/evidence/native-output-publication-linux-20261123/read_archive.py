#!/usr/bin/env python3
"""I independently check my publication controls and retain the failed neighbor."""
from pathlib import Path
import ast
import gzip
import hashlib
import json
import subprocess

BASE = Path(__file__).resolve().parent
manifest = json.loads((BASE / 'manifest.json').read_text())
hash_bytes = lambda data: hashlib.sha256(data).hexdigest()
data = {}
images = set()
for row in manifest['files']:
    compressed = (BASE / row['artifact']).read_bytes()
    assert hash_bytes(compressed) == row['compressed_sha256'], row['path']
    raw = gzip.decompress(compressed)
    assert len(raw) == row['bytes'] and hash_bytes(raw) == row['sha256'], row['path']
    assert row['path'] not in data
    data[row['path']] = raw
    if raw.startswith(b'\x7fELF'):
        assert int.from_bytes(raw[18:20], 'little') == 183
        if int.from_bytes(raw[16:18], 'little') in (2, 3):
            images.add(hash_bytes(raw))
assert {p.resolve() for p in (BASE / 'objects').glob('*.gz')} == {
    (BASE / r['artifact']).resolve() for r in manifest['files'] if r['artifact'].startswith('objects/')}
previous_path = BASE / manifest['base_manifest']['path']
assert hash_bytes(previous_path.read_bytes()) == manifest['base_manifest']['sha256']
previous = json.loads(previous_path.read_text())
old = {r['path']: r['sha256'] for r in previous['git_sources']}
current = {}
proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=BASE,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for row in manifest['git_sources']:
    proc.stdin.write((manifest['source_pin'] + ':' + row['path'] + '\n').encode())
    proc.stdin.flush()
    header = proc.stdout.readline().split()
    assert header[1] == b'blob'
    raw = proc.stdout.read(int(header[2]))
    assert proc.stdout.read(1) == b'\n'
    assert raw == data['source/' + row['path']]
    assert hash_bytes(raw) == row['sha256']
    current[row['path']] = row['sha256']
    if row['path'] not in old:
        proc.stdin.write((previous['source_pin'] + ':' + row['path'] + '\n').encode())
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        if header[1] == b'blob':
            earlier = proc.stdout.read(int(header[2]))
            assert proc.stdout.read(1) == b'\n'
            old[row['path']] = hash_bytes(earlier)
        else:
            assert header[1] == b'missing'
proc.stdin.close()
assert proc.wait() == 0
changed = {'src/main.c', 'Makefile.gnu', 'docs/ROADMAP.md', 'docs/NATIVE_PUBLICATION.md',
           'tests/test_cseed_native_publication.py'}
assert {p for p, h in current.items() if old.get(p) != h} == changed
for path, digest in old.items():
    if path not in changed:
        assert current[path] == digest, path
folder = 'journal/native-output-publication-20261123/'
def raw(path):
    return data[folder + path]
def js(path):
    return json.loads(raw(path + '.json'))
def provider(path):
    marker = '/.test_output/'
    key = 'journal/' + path.split(marker, 1)[1] if marker in path else ''
    return data[key] if key in data else data['provider/' + path]

build = js('build')
assert build['source']['src/main.c'] == current['src/main.c']
assert len(build['commands']) == 5 and all(r['exit'] == 0 for r in build['commands'])
assert set(build['modes']) == {'normal', 'san', 'production'}
for mode, plan in build['modes'].items():
    assert '-Werror' in plan['flags'] and '-std=c99' in plan['flags']
    assert ('-O3' if mode == 'production' else '-O1') in plan['flags']
    if mode == 'san':
        assert '-fsanitize=address,undefined' in plan['flags']
methods = {n.name for n in ast.walk(ast.parse(data['source/tests/test_cseed_native_publication.py']))
           if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}
assert len(methods) == 18
for mode in ('ordinary', 'private'):
    result = js(mode + '-tests')
    assert result['tests'] == 18 and result['failures'] == result['errors'] == result['skips'] == 0
    assert result['image'] == build['images'][mode]
    assert hash_bytes(provider(result['image'])) == result['sha256']
    log = raw(mode + '-tests.log').decode()
    assert 'Ran 18 tests' in log and log.endswith('\nOK\n')
    for method in methods:
        assert method + ' (tests.test_cseed_native_publication.CSeedNativePublication.' + method + ') ... ok' in log
commands = js('commands')
compile_rows = [r for r in commands if r['args'][0] in build['images'].values()]
native_rows = [r for r in commands if r not in compile_rows]
assert len(compile_rows) == 52 and len(native_rows) == 12
assert all(r['timeout'] == 120 and r['exit'] in (0, 1) for r in compile_rows)
assert all(r['timeout'] == 30 and r['exit'] == 0 for r in native_rows)
assert all(provider(r['args'][0]).startswith(b'\x7fELF') for r in native_rows)
red = js('red')
assert red['exit'] == 1 and not red['output_exists']
assert red['starts'] == red['completions'] == 1 and red['failures'] == 0
assert red['before_output_sha256'] == hash_bytes(b'I retain my previous native output.\n')
assert hash_bytes(provider(red['args'][0])) == red['compiler_sha256']
partial = js('partial-import-diagnostic')
assert partial['exit'] == 1 and not partial['source_preserved']
assert partial['source_before'] != partial['source_after']
paths = js('path-controls')['commands']
assert len(paths) == 7
alias = next(r for r in paths if r['case'] == 'source-alias')
assert alias['exit'] == 0 and alias['before']['source_sha256'] != alias['after']['source_sha256']
assert all(r['exit'] == 0 for r in js('suites')) and len(js('suites')) == 2
for mode in ('ordinary', 'private'):
    log = raw(mode + '-suite.stderr')
    assert b'Ran 21 tests' in log and log.endswith(b'\nOK\n')
neighbor = js('neighbors')
assert neighbor['image'] == build['images']['ordinary']
assert neighbor['tests'] == 16 and neighbor['failures'] == 1 and neighbor['errors'] == neighbor['skips'] == 0
log = raw('neighbors.log')
assert b'test_source_only_does_not_execute_dependency_shadows' in log and b'FAILED (failures=1)' in log
predecessor = js('predecessor-neighbor')
assert predecessor['tests'] == predecessor['failures'] == 1 and predecessor['errors'] == 0
refusal = b'I require the exact emitted declaration for this qualified C call.'
refusals = js('qualified-c-import')['commands']
assert {r['mode'] for r in refusals} == {'predecessor', 'candidate'}
for row in refusals:
    assert row['exit'] == 1 and row['prior_output_preserved'] and not row['shadow_report_exists']
    assert refusal in provider(str(Path(js('qualified-c-import')['work']) / (row['mode'] + '.stderr')))
assert hash_bytes(provider('/usr/bin/gcov')) == js('coverage-readers')['tool_sha256']
for row in js('coverage-readers')['commands']:
    assert row['exit'] == 0
    assert provider(row['notes']) and provider(row['data'])
    assert b'Creating' in provider(str(Path(row['work']) / 'stdout'))
assert js('make')['exit'] == 0 and b'Ran 18 tests' in raw('make.stderr')
assert raw('make.stderr').endswith(b'\nOK\n')
assert all(r['exit'] == 0 for r in js('policy'))
for path, digest in json.loads(data['baseline/product-closure.json']).items():
    assert hash_bytes(provider(path)) == digest
for path, record in json.loads(data['baseline/image-guards.json']).items():
    matches = [value for key, value in data.items() if key.startswith('provider/') and key.endswith('/' + path)]
    assert len(matches) == 1 and hash_bytes(matches[0]) == record['sha256']
a = manifest['acceptance']
assert a['focused_native_publication'] and a['source_identity_guards'] and a['original_enum_neighbors']
assert not a['original_shadow_neighbors']
assert not any(a[k] for k in ('ordinary_full_source', 'private_full_source', 'fresh_shared_provider_build',
                             'paired_platform', 'raw_stage', 'whole_release'))
print(json.dumps(dict(result='PASS', source=manifest['source_pin'], records=len(data),
                      git_inputs=len(current), native_images=len(images),
                      focused_native_publication=True, original_neighbor_suite=False,
                      full_source_acceptance=False, whole_release=False), indent=2))
