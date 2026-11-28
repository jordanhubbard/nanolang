#!/usr/bin/env python3
"""I check archived bytes, Git identities, native products and the claimed gates."""
from pathlib import Path
import ast
from collections import Counter
import gzip
import hashlib
import json
import re
import subprocess

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
MANIFEST = json.loads((BASE / 'manifest.json').read_text())
PIN = 'c93263c64f6d5dd1db0e856284d76f1c8e0fa131'
assert MANIFEST['source_pin'] == PIN
sha = lambda b: hashlib.sha256(b).hexdigest()
rows = {row['path']: row for row in MANIFEST['files']}
assert len(rows) == len(MANIFEST['files']) == 14924
images = set()
verified = {}
for row in rows.values():
    path = (BASE / row['artifact']).resolve()
    assert path.is_relative_to(REPO / 'docs/evidence'), path
    identity = (row['compressed_sha256'], row['sha256'], row['bytes'])
    if path in verified:
        assert verified[path] == identity
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == row['compressed_sha256'], row['path']
    digest = hashlib.sha256()
    size = 0
    header = b''
    with gzip.open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            if not header:
                header = block[:64]
            size += len(block)
            digest.update(block)
    assert size == row['bytes'] and digest.hexdigest() == row['sha256'], row['path']
    # Deliberately malformed ELF fixtures are source inputs, not native images.
    if (header[:6] == b'\x7fELF\x02\x01' and len(header) == 64
            and int.from_bytes(header[16:18], 'little') in (2, 3)
            and int.from_bytes(header[18:20], 'little') == 183):
        images.add(row['sha256'])
    verified[path] = identity
assert {p.resolve() for p in (BASE / 'objects').glob('*.gz')} == {
    (BASE / r['artifact']).resolve() for r in rows.values()
    if r['artifact'].startswith('objects/')}
prior_path = BASE / MANIFEST['base_manifest']['path']
assert sha(prior_path.read_bytes()) == MANIFEST['base_manifest']['sha256']
prior = json.loads(prior_path.read_text())

def data(key):
    row = rows[key]
    return gzip.decompress((BASE / row['artifact']).read_bytes())

def raw(name):
    return data('journal/source-c-imports-20261123/' + name)

def js(name):
    return json.loads(raw(name + '.json'))

def provider_key(path):
    key = 'journal/' + path.split('/.test_output/', 1)[1] if '/.test_output/' in path else ''
    return key if key in rows else 'provider/' + path

def provider(path):
    return data(provider_key(path))

def native(path):
    row = rows[provider_key(path)]
    assert row['sha256'] in images, path
    return row

# Compare every archived repository source with its actual Git blob.
proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def blob(pin, name):
    proc.stdin.write((pin + ':' + name + '\n').encode())
    proc.stdin.flush()
    header = proc.stdout.readline().split()
    if header[1] == b'missing':
        return None
    assert header[1] == b'blob'
    result = proc.stdout.read(int(header[2]))
    assert proc.stdout.read(1) == b'\n'
    return result
current = {}
old = {r['path']: r['sha256'] for r in prior['git_sources']}
for row in MANIFEST['git_sources']:
    name = row['path']
    value = blob(PIN, name)
    assert value is not None and len(value) == row['bytes']
    assert sha(value) == row['sha256'] == rows['source/' + name]['sha256'], name
    current[name] = row['sha256']
    if name not in old:
        previous = blob(prior['source_pin'], name)
        if previous is not None:
            old[name] = sha(previous)
proc.stdin.close()
assert proc.wait() == 0 and len(current) == 3518
changed = set(subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', PIN],
                                      cwd=REPO, text=True).splitlines())
assert len(changed) == 19
assert {p for p, digest in current.items() if old.get(p) != digest} == changed
# Original Python assertion calls remain exact except two documented spelling oracles.
assertion_changes = set()
for name in current:
    if not re.fullmatch(r'tests/test_public_c_[^/]+\.py', name):
        continue
    previous = subprocess.check_output(['git', 'show', prior['source_pin'] + ':' + name], cwd=REPO)
    def assertions(source):
        tree = ast.parse(source)
        return {n.name: Counter(ast.dump(c) for c in ast.walk(n)
                               if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                               and c.func.attr.startswith('assert'))
                for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}
    earlier, later = assertions(previous), assertions(data('source/' + name))
    assert earlier.keys() == later.keys(), name
    for method in earlier:
        if earlier[method] != later[method]:
            assertion_changes.add((name, method))
assert assertion_changes == {
    ('tests/test_public_c_bool_text.py', 'test_helper_names_once_effects_and_stable_aliases'),
    ('tests/test_public_c_union_results.py', 'test_scalar_payload_alias_parameter_local_and_forward_returns')}
wiring = json.loads((BASE / 'consumer-wiring.json').read_text())
compressed_make = (BASE / 'consumer-Makefile.gnu.gz').read_bytes()
assert sha(compressed_make) == wiring['compressed_sha256']
consumer_make = gzip.decompress(compressed_make)
assert sha(consumer_make) == wiring['makefile_sha256']
assert consumer_make == data('source/Makefile.gnu').replace(
    b'test-units: test-source-c-graph-api\n', b'test-units: test-source-c-imports\n')
assert b'test-source-c-imports: $(COMPILER_C) test-source-c-graph-api\n' in consumer_make
assert wiring['tested_target_exit'] == 0 and wiring['source_pin'] == PIN
assert all(current[path] == digest for path, digest in wiring['source_sha256'].items())

frozen = js('frozen-source-before')
assert frozen == js('frozen-source-after') and frozen['pin'] == PIN
assert len(frozen['files']) == 3500
assert all(current[path] == digest for path, digest in frozen['files'].items())
tools = js('compiler-tools-before')
assert tools['files'] == js('compiler-tools-after')['files'] and len(tools['files']) == 24
assert all(r['exit'] == 0 and not r['stderr'] for r in tools['reports'].values())
for path, digest in tools['files'].items():
    assert native(path)['sha256'] == digest
public = json.loads(data('baseline/image-guards.json'))
libraries = json.loads(data('baseline/product-closure.json'))
assert js('final-product-guards') == dict(public=public, libraries=libraries)
assert len(public) == 5 and len(libraries) == 9
for path, row in public.items():
    matches = [r for k, r in rows.items() if k.startswith('provider/') and k.endswith('/' + path)]
    assert len(matches) == 1 and matches[0]['sha256'] == row['sha256']
for path, digest in libraries.items():
    assert native(path)['sha256'] == digest

build = js('build')
assert len(build['commands']) == 8 and all(r['exit'] == 0 for r in build['commands'])
assert build['source'] == {p: current[p] for p in build['source']}
assert set(build['source']) == {'src/main.c', 'src/c_backend.c', 'src/c_backend.h', 'src/c_backend_values.inc'}
assert set(build['modes']) == {'normal', 'san', 'production'}
for mode, plan in build['modes'].items():
    assert {'-Werror', '-Wall', '-Wextra', '-std=c99'} <= set(plan['flags'])
    assert ('-O3' if mode == 'production' else '-O1') in plan['flags']
    if mode == 'san':
        assert '-fsanitize=address,undefined' in plan['flags']
    for path in plan['objects'].values():
        assert provider(path).startswith(b'\x7fELF')
for path in build['images'].values():
    native(path)

def methods(path):
    return {n.name for n in ast.walk(ast.parse(data('source/' + path)))
            if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}

def passed(log, count, names=()):
    text = log.decode()
    assert re.search(r'Ran ' + str(count) + r' tests? in ', text)
    assert text.endswith('\nOK\n'), text[-500:]
    for name in names:
        assert re.search(r'^' + re.escape(name) + r' \([^\n]+\) \.\.\. ok$', text, re.M), name

import_methods = methods('tests/test_source_c_imports.py')
assert len(import_methods) == 20
focused = js('focused')
assert {r['mode'] for r in focused} == {'ordinary', 'private'}
for row in focused:
    assert row['exit'] == 0 and row['image'] == build['images'][row['mode']]
    passed(raw(row['mode'] + '-focused.stderr'), 20, import_methods)
api = js('api-controls')
assert len(api['commands']) == 18
for row in api['commands']:
    expected = 42 if re.fullmatch(r'(normal|san|production)-c(99|11)', row['label']) else 0
    assert row['exit'] == expected
    assert provider(api['work'] + '/' + row['label'] + '.stderr') == b'' or '-api' in row['label']
    if row['label'].endswith('-api'):
        assert b'I preserve output through 6 allocation failures.' in provider(api['work'] + '/' + row['label'] + '.stdout')
        native(row['args'][0])
san_binary = provider(api['work'] + '/san-api')
assert b'__asan_init' in san_binary and b'__ubsan_handle' in san_binary
assert b'detect_leaks=1:halt_on_error=1' in raw('api-controls.py')
assert b"UBSAN_OPTIONS='halt_on_error=1'" in raw('api-controls.py')

clang = js('clang-consumers')
assert len(clang) == 207 and all(r['exit'] == 0 for r in clang[:3])
assert [r['args'][0] for r in clang[:3]] == ['/usr/bin/cc', '/bin/clang-18', '/usr/bin/as']
for index in range(3, len(clang), 3):
    compile_row, reference, actual = clang[index:index + 3]
    assert compile_row['label'] == 'compile' and compile_row['exit'] == 0
    assert compile_row['args'][0] == '/bin/clang-18'
    assert {'-pedantic-errors', '-Wall', '-Wextra', '-Werror'} <= set(compile_row['args'])
    assert '-std=c99' in compile_row['args'] or '-std=c11' in compile_row['args']
    assert provider(compile_row['args'][6])  # The exact already-GCC-consumed C input.
    assert reference['label'] == 'gcc-reference' and actual['label'] == 'clang-product'
    assert reference['exit'] == actual['exit']
    native(reference['args'][0]); native(actual['args'][0])
    for suffix in ('stdout', 'stderr'):
        assert raw('clang-%03d.%s' % (index + 1, suffix)) == raw('clang-%03d.%s' % (index + 2, suffix))
public_result = js('public-c-controls')
assert (public_result['tests'], public_result['failures'], public_result['errors']) == (66, 0, 0)
assert len(public_result['commands']) == 685
passed(raw('public-c-controls.stderr'), 66)
assert len([p for p in current if re.fullmatch(r'tests/test_public_c_[^/]+\.py', p)]) == 17
assert public_result['image'] == build['images']['ordinary']
assert public_result['commands'] == js('public-c-commands')
assert all(r['timeout'] == 240 for r in public_result['commands'])
assert any('-fsanitize=address,undefined' in r['args'] for r in public_result['commands'])
for i, row in enumerate(public_result['commands']):
    for suffix in ('stdout', 'stderr'):
        provider(public_result['work'] + '/' + str(i) + '.' + suffix)
neighbor = js('neighbors')
assert (neighbor['tests'], neighbor['failures'], neighbor['errors'], neighbor['skips']) == (16, 0, 0, 0)
passed(raw('neighbors.log'), 16)
publication_methods = methods('tests/test_cseed_native_publication.py')
assert len(publication_methods) == 18
for mode in ('ordinary', 'private'):
    result = js(mode + '-tests')
    assert (result['tests'], result['failures'], result['errors'], result['skips']) == (18, 0, 0, 0)
    assert result['image'] == build['images'][mode]
    assert native(result['image'])['sha256'] == result['sha256']
    passed(raw(mode + '-tests.log'), 18, publication_methods)
    passed(raw(mode + '-suite.stderr'), 21, methods('tests/test_cseed_enum_lists.py'))
assert len(js('suites')) == 2 and all(r['exit'] == 0 for r in js('suites'))
assert len(js('policy')) == 2 and all(r['exit'] == 0 for r in js('policy'))
assert js('make')['exit'] == 0 and js('make')['args'][-1] == 'test-source-c-imports'
passed(raw('make.stderr'), 20)
assert b'I preserve output through 6 allocation failures.' in raw('make.stdout')
for name, count in [('complete-qualification', 3), ('initial-gates', 2), ('final-gates', 7)]:
    assert len(js(name)) == count and all(r['exit'] == 0 for r in js(name))

# Keep unsuccessful observations visible; a later passing subset does not erase them.
for directory, tests, failures in [('first-15-failures', 15, 2), ('first-17-union-refusal', 17, 1),
                                    ('first-20-fixture-parse-refusal', 20, 1), ('first-20-unused-payload-refusal', 20, 1)]:
    for mode in ('ordinary', 'private'):
        log = raw(directory + '/' + mode + '-focused.stderr')
        assert ('Ran %d tests' % tests).encode() in log
        assert ('FAILED (failures=%d)' % failures).encode() in log
assert b'13 errors generated.' in raw('first-clang-unused-refusal/clang-003.stderr')
baseline = js('clang-baseline')
assert baseline['exit'] == 1
assert sha(provider(baseline['args'][6])) == baseline['source_sha256']
assert b'13 errors generated.' in provider(baseline['work'] + '/stderr')
for name in ('first-public-66-failures', 'public-66-named-failures'):
    red = js(name + '/public-c-controls')
    assert (red['tests'], red['failures'], red['errors']) == (66, 6, 4)
    assert len(red['commands']) == 657
assert b'FAILED (failures=6, errors=4)' in raw('public-66-named-failures/public-c-controls.stderr')
setup = js('targeted-observer-name-setup/fixture-recovery-targeted')
assert (setup['tests'], setup['failures'], setup['errors']) == (10, 0, 10) and not setup['commands']
targeted = js('fixture-recovery-targeted')
assert (targeted['tests'], targeted['failures'], targeted['errors']) == (10, 0, 0)
predecessor = js('fixture-predecessor')
assert len(predecessor['commands']) == 9
for row in predecessor['commands']:
    assert row['exit'] == (1 if row['label'].endswith('-missing') else 0)
order = js('argument-order')['commands']
assert all(r['exit'] == (12 if r['label'].endswith('-run') or r['label'] == 'ordinary-native' else 0) for r in order)
acceptance = MANIFEST['acceptance']
true_keys = {'checked_source_c_graph', 'ordered_source_c_calls', 'strict_gcc_clang_consumers',
             'complete_public_c_profile', 'original_shadow_neighbors', 'original_enum_neighbors', 'native_publication_neighbors'}
assert {key for key, value in acceptance.items() if value} == true_keys
assert set(acceptance) - true_keys == {'ordinary_full_source', 'private_full_source', 'fresh_shared_provider_build',
                                      'full_instrumented_driver', 'paired_platform', 'raw_stage', 'whole_release'}
print(json.dumps(dict(result='PASS', source=PIN, records=len(rows), git_inputs=len(current),
                      unique_artifacts=len(verified), native_images=len(images), frozen_inputs=3500,
                      public_c_methods=66, public_c_commands=685, strict_consumer_commands=len(clang),
                      full_source_acceptance=False, paired_platform=False, whole_release=False), indent=2))
