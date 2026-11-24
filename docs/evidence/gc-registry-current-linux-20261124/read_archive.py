#!/usr/bin/env python3
"""I check atomic GC refusal and preserve the failed predecessor."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import re
import struct
import subprocess

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
m = json.loads((BASE / 'manifest.json').read_text())
sha = lambda b: hashlib.sha256(b).hexdigest()
rows = {r['path']: r for r in m['files']}
assert len(rows) == len(m['files'])
verified = {}
images = set()
for row in rows.values():
    path = (BASE / row['artifact']).resolve()
    assert path.is_relative_to(REPO / 'docs/evidence')
    identity = row['compressed_sha256'], row['sha256'], row['bytes']
    if path in verified:
        assert verified[path] == identity
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == row['compressed_sha256']
    digest = hashlib.sha256(); size = 0; head = b''
    with gzip.open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            if not head: head = block[:64]
            digest.update(block); size += len(block)
    assert digest.hexdigest() == row['sha256'] and size == row['bytes']
    if (head[:6] == b'\x7fELF\x02\x01' and len(head) == 64
            and int.from_bytes(head[16:18], 'little') in (2, 3)
            and int.from_bytes(head[18:20], 'little') == 183):
        images.add(row['sha256'])
    verified[path] = identity
assert {p.resolve() for p in (BASE / 'objects').glob('*.gz')} == {
    (BASE / row['artifact']).resolve() for row in rows.values() if row['artifact'].startswith('objects/')}
prior_path = BASE / m['base_manifest']['path']
assert sha(prior_path.read_bytes()) == m['base_manifest']['sha256']

def data(name):
    return gzip.decompress((BASE / rows[name]['artifact']).read_bytes())

def raw(name):
    return data('journal/gc-registry-current-20261124/' + name)

def js(name):
    return json.loads(raw(name + '.json'))

def provider(path):
    key = 'journal/' + path.split('/.test_output/', 1)[1] if '/.test_output/' in path else ''
    return data(key if key in rows else 'provider/' + path)

pin = '8d970c29319878fbd27b95349a2ed43b10759b1c'
assert m['source_pin'] == pin
proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
current = {}
for row in m['git_sources']:
    proc.stdin.write((pin + ':' + row['path'] + '\n').encode()); proc.stdin.flush()
    header = proc.stdout.readline().split(); assert header[1] == b'blob'
    value = proc.stdout.read(int(header[2])); assert proc.stdout.read(1) == b'\n'
    assert sha(value) == row['sha256'] == rows['source/' + row['path']]['sha256']
    assert len(value) == row['bytes']; current[row['path']] = row['sha256']
proc.stdin.close(); assert proc.wait() == 0 and len(current) == 3519
previous = json.loads(prior_path.read_text())
old = {r['path']: r['sha256'] for r in previous['git_sources']}
assert {p for p, digest in current.items() if old.get(p) != digest} == {
    'src/runtime/gc.c', 'tests/test_gc_allocation_failure.c', 'Makefile.gnu', 'docs/ROADMAP.md'}
for name in ['src/runtime/gc.c', 'tests/test_gc_allocation_failure.c']:
    origin = subprocess.check_output(['git', 'show', m['reviewed_origin'] + ':' + name], cwd=REPO)
    assert sha(origin) == current[name]
assert current['src/runtime/gc.h'] == old['src/runtime/gc.h']
red = js('red')
assert len(red['commands']) == 2
assert [r['exit'] for r in red['commands']] == [0, 1]
assert red['gc_sha256'] == old['src/runtime/gc.c'] == sha(raw('gc-before.c'))
assert red['test_sha256'] == current['tests/test_gc_allocation_failure.c']
assert b'owner == NULL' in provider(red['work'] + '/run.stderr')
assert provider(red['work'] + '/gc-before.c') == raw('gc-before.c')
assert provider(red['work'] + '/test.c') == data('source/tests/test_gc_allocation_failure.c').replace(
    b'#include "../src/runtime/gc.c"',
    ('#include "' + str(REPO / 'src/runtime/gc.c') + '"').encode())
assert '-fsanitize=address,undefined' in red['commands'][0]['args']
frozen = js('frozen-source')
assert frozen['pin'] == pin and len(frozen['files']) == 3501
assert all(current[path] == digest for path, digest in frozen['files'].items())
qualified = js('qualified'); assert qualified['before'] == qualified['after']
assert all(current[path] == digest for path, digest in qualified['before'].items())
assert len(qualified['commands']) == 13 and all(r['exit'] == 0 for r in qualified['commands'])
clang = js('clang-controls')
assert len(clang['commands']) == 12 and all(r['exit'] == 0 for r in clang['commands'])
for result in [qualified, clang]:
    for row in result['commands']:
        if row['label'] == 'original-make': continue
        if row['label'].endswith('-build'):
            assert {'-Wall', '-Wextra', '-Werror', '-std=c99'} <= set(row['args'])
            if row['label'].startswith('san-'):
                assert '-fsanitize=address,undefined' in row['args']
                assert '-fno-sanitize-recover=all' in row['args']
        elif row['label'].endswith('-allocation'):
            assert b'I passed header/registry refusal, three constructor families, retries, finalizers and reinitialization.' in provider(result['work'] + '/' + row['label'] + '.stdout')
        else:
            assert b'All 13 gc_struct tests passed.' in provider(result['work'] + '/' + row['label'] + '.stdout')
make = next(row for row in qualified['commands'] if row['label'] == 'original-make')
assert make['args'][-4:] == ['test-gc-shutdown-sanitizers', 'test-eval-bstring-sanitizers',
                            'test-eval-record-fields-sanitizers', 'test-nl-string-bounds-sanitizers']
make_output = provider(qualified['work'] + '/original-make.stdout')
assert make_output.count(b'ASAN_OPTIONS=detect_leaks=1 ') == 5
assert b'2000 actual record lookups with 2000 exact-name comparisons' in make_output
assert b'I passed header/registry refusal' in make_output and b'All 13 gc_struct tests passed.' in make_output
assert b'All tests passed!' in make_output
old_make_row = next(r for r in previous['files'] if r['path'] == 'source/Makefile.gnu')
old_make = gzip.decompress((prior_path.parent / old_make_row['artifact']).read_bytes())
marker = b'\tASAN_OPTIONS=detect_leaks=$(if $(filter Darwin,$(UNAME_S)),0,1) $(OBJ_DIR)/test_gc_shutdown\n'
addition = (b'\t$(CC) $(CFLAGS) -UNDEBUG -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer \\\n'
            b'\t\ttests/test_gc_allocation_failure.c src/runtime/dyn_array.c src/runtime/gc_struct.c \\\n'
            b'\t\t-o $(OBJ_DIR)/test_gc_allocation_failure $(LDFLAGS)\n'
            b'\tASAN_OPTIONS=detect_leaks=$(if $(filter Darwin,$(UNAME_S)),0,1) $(OBJ_DIR)/test_gc_allocation_failure\n')
assert old_make.count(marker) == 1
assert data('source/Makefile.gnu') == old_make.replace(marker, marker + addition)
neighbor = js('neighbors')
assert len(neighbor['commands']) == 13 and all(row['exit'] == 0 for row in neighbor['commands'])
assert all(current[path] == digest for path, digest in neighbor['source'].items())
for path, digest in neighbor['providers'].items():
    assert sha(provider(path)) == digest
for mode in ['normal', 'san', 'production']:
    assert b'=== 60 passed, 0 failed ===' in provider(neighbor['work'] + '/' + mode + '-environment.stdout')
lexical = neighbor['lexical']
assert (lexical['tests'], lexical['failures'], lexical['errors']) == (10, 0, 0)
assert provider(neighbor['work'] + '/lexical-tests.log').endswith(b'\nOK\n')
assert b'Ran 10 tests' in provider(neighbor['work'] + '/lexical-tests.log')
assert all(row['timeout'] in (10, 120) for row in lexical['commands'])
assert any(row['args'][0] == neighbor['compiler'] for row in lexical['commands'])
assert sha(provider(neighbor['compiler'])) in images
for root in [qualified['work'], clang['work']]:
    for name in ['san-allocation', 'san-original']:
        binary = provider(root + '/' + name)
        assert b'__asan_init' in binary and b'__ubsan_handle' in binary
        assert sha(binary) in images
assert b'detect_leaks=1:halt_on_error=1' in raw('qualify.py')
assert b'detect_leaks=1:halt_on_error=1' in raw('clang-controls.py')
tools = js('tool-guard-continuity')
assert tools['before'] == tools['after'] and len(tools['before']) == 24
for path, digest in tools['after'].items():
    assert sha(provider(path)) == digest
products = js('public-product-guards')
assert len(products['public']) == 5 and len(products['libraries']) == 9
for name, row in products['public'].items():
    matches = [r for key, r in rows.items() if key.startswith('provider/') and key.endswith('/' + name)]
    assert len(matches) == 1 and matches[0]['sha256'] == row['sha256']
for path, digest in products['libraries'].items():
    assert sha(provider(path)) == digest
assert {key for key, value in m['acceptance'].items() if value} == {
    'atomic_registry_refusal', 'original_gc_controls', 'original_environment_controls', 'original_string_controls'}
print(json.dumps(dict(result='PASS', source=pin, reviewed_origin=m['reviewed_origin'],
                      records=len(rows), git_inputs=len(current), native_images=len(images),
                      frozen_inputs=3501, gcc_matrix_commands=12, clang_matrix_commands=12,
                      original_gc_methods=13, environment_checks_per_profile=60,
                      lexical_methods=10, full_source_acceptance=False, paired_platform=False,
                      default_ownership_enabled=False, whole_release=False), indent=2))
