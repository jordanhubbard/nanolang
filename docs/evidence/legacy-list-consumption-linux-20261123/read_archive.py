#!/usr/bin/env python3
"""I verify immediate legacy-list consumption and retain my remaining release gates."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess

base = Path(__file__).resolve().parent
m = json.loads((base / 'manifest.json').read_text())
h = lambda b: hashlib.sha256(b).hexdigest()
priorpath = base / m['base_manifest']['path']
priorraw = priorpath.read_bytes()
assert h(priorraw) == m['base_manifest']['sha256']
prior = json.loads(priorraw)
rows = {}
images = set()
for r in m['files']:
    z = (base / r['artifact']).read_bytes()
    assert h(z) == r['compressed_sha256'], r['path']
    b = gzip.decompress(z)
    assert len(b) == r['bytes'] and h(b) == r['sha256'], r['path']
    assert r['path'] not in rows
    rows[r['path']] = b
    if b.startswith(b'\x7fELF') and int.from_bytes(b[16:18], 'little') in (2, 3):
        assert int.from_bytes(b[18:20], 'little') == 183
        images.add(h(b))
assert {p.resolve() for p in (base / 'objects').glob('*.gz')} == {
    (base / r['artifact']).resolve() for r in m['files'] if r['artifact'].startswith('objects/')}

def prior_record(name):
    r = next(r for r in prior['files'] if r['path'] == name)
    z = (priorpath.parent / r['artifact']).read_bytes()
    assert h(z) == r['compressed_sha256']
    b = gzip.decompress(z)
    assert h(b) == r['sha256'] and len(b) == r['bytes']
    return b

def js(name):
    return json.loads(rows[name + '.json'])

proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=base,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
new = {}
for r in m['git_sources']:
    proc.stdin.write((m['source_pin'] + ':' + r['path'] + '\n').encode())
    proc.stdin.flush()
    header = proc.stdout.readline().split()
    assert header[1] == b'blob'
    b = proc.stdout.read(int(header[2]))
    assert proc.stdout.read(1) == b'\n'
    assert h(b) == r['sha256'] and b == rows['source/' + r['path']]
    new[r['path']] = r['sha256']
proc.stdin.close()
assert proc.wait() == 0
old = {r['path']: r['sha256'] for r in prior['git_sources']}
assert {p for p, v in new.items() if old.get(p) != v} == {
    'src/eval.c', 'src/eval/carrier_heap.c', 'src/eval/carrier_heap.h',
    'src/eval/carrier_values.c', 'src/eval/carrier_values.h',
    'src/eval/carrier_runtime.c', 'src/eval/carrier_runtime.h',
    'Makefile.gnu', 'docs/ROADMAP.md', 'docs/EVAL_CARRIER_OWNERSHIP.md',
    'tests/test_eval_carrier_heap.c', 'tests/test_eval_carrier_values.c',
    'tests/test_eval_legacy_list_free.c', 'tests/test_carrier_legacy_list_free.c'}
for p in old:
    if p.startswith('src/') and p not in {
        'src/eval.c', 'src/eval/carrier_heap.c', 'src/eval/carrier_heap.h',
        'src/eval/carrier_runtime.c', 'src/eval/carrier_runtime.h',
        'src/eval/carrier_values.c', 'src/eval/carrier_values.h'}:
        assert new[p] == old[p], p
# I preserve all original engine and adapter assertions.
engine = rows['source/tests/test_eval_carrier_heap.c'].decode()
a = engine.index('static void immediate_destruction(')
b = engine.index('int main(', a)
engine = (engine[:a] + engine[b:]).replace('    immediate_destruction();\n', '')
engine = engine.replace('    CHECK(!nl_carrier_heap_destroy(ctx->heap, value));\n', '')
assert engine.encode() == prior_record('source/tests/test_eval_carrier_heap.c')
values = rows['source/tests/test_eval_carrier_values.c'].decode()
a = values.index('static void consumption_allocation_rollback(')
b = values.index('int main(', a)
values = (values[:a] + values[b:]).replace('    consumption_allocation_rollback();\n', '')
assert values.encode() == prior_record('source/tests/test_eval_carrier_values.c')
assert rows['journal/original-explicit-free-probe.c'] == prior_record('journal/explicit-free-probe.c')
assert b'heap-use-after-free' in prior_record('journal/explicit-free-run.stderr')

def provider(path):
    marker = '/.test_output/legacy-list-consumption-20261123/'
    return rows['journal/' + path.split(marker, 1)[1]] if marker in path else rows['provider/' + path]

for p, digest in js('journal/qualified-source').items():
    assert new[p] == digest, p
repair = js('journal/repair')
assert len(repair['commands']) == 24 and all(c['exit'] == 0 for c in repair['commands'])
assert set(repair['modes']) == {'normal', 'san', 'production'}
for p, digest in repair['providers'].items():
    assert h(provider(p)) == digest
    assert any(r['sha256'] == digest for r in prior['files']), p
for mode in ['normal', 'san', 'production']:
    for subject in ['env', 'eval', 'repl', 'carrier_heap', 'carrier_values', 'carrier_runtime']:
        a = next(c['args'] for c in repair['commands'] if c['label'] == mode + '-compile-' + subject)
        assert '-Werror' in a and ('-O3' if mode == 'production' else '-O1') in a
        if mode == 'san': assert '-fsanitize=address,undefined' in a
for subject, count in [('controls', 42), ('neighbors', 76), ('extra', 19)]:
    commands = js('journal/' + subject)
    assert len(commands) == count and all(c['exit'] == 0 for c in commands)
for mode in ['normal', 'san', 'production']:
    assert b'I preserve lower-case record-list shadows' in rows['journal/' + mode + '-original-free-run.stdout']
    assert b'AddressSanitizer:' not in rows['journal/' + mode + '-original-free-run.stderr']
    assert b'runtime error:' not in rows['journal/' + mode + '-original-free-run.stderr']
    for subject, count in [('default34', 34), ('private41', 41), ('current-private41', 41)]:
        err = rows['journal/' + mode + '-' + subject + '.stderr']
        assert ('Ran ' + str(count) + ' tests').encode() in err and b'\nOK\n' in err
        assert b'AddressSanitizer:' not in err and b'runtime error:' not in err
for mode in ['normal', 'san']:
    assert rows['journal/' + mode + '-eval-run.stdout'].count(b'  Testing ') == 134
    assert b'All eval tests passed!' in rows['journal/' + mode + '-eval-run.stdout']
    assert rows['journal/' + mode + '-checker-run.stdout'].count(b'  Testing ') == 88
    assert b'All typechecker tests passed!' in rows['journal/' + mode + '-checker-run.stdout']
    assert b'delta=0 calls=128.' in rows['journal/' + mode + '-native-bytes-run.stdout']
final = js('journal/final-controls')
assert len(final) == 15
assert all(c['exit'] == (1 if c['label'] == 'san-default-admission' else 0) for c in final)
import re
for mode in ['normal', 'san', 'production']:
    make = next(c for c in final if c['label'] == mode + '-final-make')
    assert set(make['args']) >= {'test-eval-legacy-record-lists', 'test-carrier-legacy-list-values',
                               'test-eval-legacy-list-free', 'test-carrier-legacy-list-free'}
    assert len(make['products']) == 4
    for path, digest in make['products'].items(): assert h(provider(path)) == digest
    out = rows['journal/' + mode + '-final-make.stdout']
    assert b'I refuse scalar bits, malformed headers, wrong managed layouts and owned nonlist storage' in out
    assert b'I preserve immediate free, copied STRING/record values' in out
    match = re.search(rb'positive=(\d+) after128=(\d+) after512=(\d+) delta=(-?\d+) rounds=512', out)
    assert match
    positive, middle, last, delta = map(int, match.groups())
    assert positive >= 2 * 1024 * 1024 and last <= middle + 1024 and delta == last - middle
    assert b'AddressSanitizer:' not in rows['journal/' + mode + '-final-make.stderr']
    assert b'Ran 7 tests' in rows['journal/' + mode + '-default-admission.stderr']
err = rows['journal/san-default-admission.stderr']
assert b'FAILED (failures=3)' in err
assert err.count(b'168 byte(s) leaked in 6 allocation(s)') == 1
assert err.count(b'172 byte(s) leaked in 6 allocation(s)') == 2
# My final Make gates include the additional refusal tests. Earlier focused
# controls and the predecessor's ordinary stage remain separate observations.
assert b'__asan_poison_memory_region' in rows['source/tests/test_carrier_legacy_list_free.c']
assert b'free(list->data)' in rows['source/tests/test_carrier_legacy_list_free.c']
assert b'nl_carrier_values_begin_list_borrow' in rows['source/tests/test_carrier_legacy_list_free.c']
assert b'-Wl,--wrap' not in rows['source/Makefile.gnu']
predecessor = js('journal/legacy-only-borrow/ordinary-qualified')
assert (predecessor['exit'], predecessor['starts'], predecessor['completions']) == (0, 1306, 1306)
assert h(rows['journal/legacy-only-borrow/reconstructed-compiler']) == predecessor['image_sha256']
assert js('journal/legacy-only-borrow/reconstruction')['full_source_repeated'] is False
for path, digest in predecessor['providers'].items(): assert h(provider(path)) == digest
q = js('journal/ordinary-qualified')
assert (q['exit'], q['starts'], q['completions']) == (0, 1306, 1306)
assert q['before'] == q['after'] and len(q['before']) == 1340
assert q['timeout'] == 600 and q['shadow_timeout'] == 'default10' and not q['timed_out']
for p, digest in q['after'].items(): assert new[p] == digest, p
assert h(provider(q['args'][0])) == q['image_sha256']
for p, digest in q['providers'].items(): assert h(provider(p)) == digest
observed = []
for line in rows['journal/ordinary-native/stderr'].splitlines():
    try:
        value = json.loads(line)
        if isinstance(value, dict): observed.append(value)
    except ValueError: pass
starts = [v for v in observed if v.get('event') == 'shadow-start']
ends = [v for v in observed if v.get('event') == 'shadow-complete']
assert len(starts) == len(ends) == 1306 and all(v['failures'] == 0 for v in ends)
assert [v for v in observed if v.get('event') == 'cseed-shadow-phase'] == q['phases']
status = q['phases'][-1]
assert status['reaped'] and status['completed'] and not status['parent_timeout']
assert status['signal'] == status['exit_code'] == 0 and status['deadline_seconds'] == 10
assert h(rows['journal/ordinary-native/seed']) == q['product']['sha256']
assert js('journal/ordinary-seed-help')['exit'] == 0
assert js('journal/ordinary-seed-help')['sha256'] == q['product']['sha256']
for subject, count in [('lexical', 10), ('record-payload', 7)]:
    result = js('journal/' + subject + '-qualified')
    assert (result['tests'], result['failures'], result['errors'], result['success']) == (count, 0, 0, True)
    commands = js('journal/' + subject + '-commands')
    expected = [1 if i in {2, 3, 4, 5, 6, 23} else 0 for i in range(24)] if subject == 'lexical' else [0] * 12 + [1] * 3
    assert [c['exit'] for c in commands] == expected
    for c in commands:
        assert c['shadow_timeout'] is None
        for item in c['inputs']:
            assert h(rows['journal/' + subject + '-inputs/' + item['sha256']]) == item['sha256']
assert all(c['exit'] == 0 for c in js('journal/policy'))
assert js('journal/final-image-guards') == json.loads(rows['baseline/image-guards.json'])
closure = js('journal/product-closure')
assert len(closure) == 9
for p, digest in closure.items(): assert h(provider(p)) == digest
assert m['acceptance'] == dict(
    explicit_legacy_free=True, immediate_native_destructors=True, borrowed_fresh_lifetimes=True,
    deterministic_reuse_model=True, focused_legacy_lsan=True, ordinary_compiler_source=True,
    default_repl34=True, private_repl41=True, full_private_compiler_source=False,
    default_aggregate_ownership=False, fresh_shared_provider_build=False,
    paired_platform=False, raw_stage=False, whole_release=False)
print(f'PASS {len(rows)} records / {len(new)} exact Git inputs / {len(images)} native ELF images; '
      'original free probe and 512-round native controls pass across three profiles; '
      'ordinary1306 complete; private cost/default/raw/paired/release open')
