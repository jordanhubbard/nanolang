#!/usr/bin/env python3
"""I verify legacy root provenance and retain my incomplete private gates."""
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
assert m['source_pin'] == 'ccacf817baadcfaf98d1962c7dcc012e10fd599f'
assert {p for p, v in new.items() if old.get(p) != v} == {
    'src/eval.c', 'src/eval/carrier_values.c', 'src/eval/carrier_values.h',
    'src/eval/carrier_runtime.c', 'src/eval/carrier_runtime.h', 'Makefile.gnu',
    'tests/test_eval_legacy_record_lists.c', 'tests/test_carrier_legacy_list_values.c',
    'docs/ROADMAP.md', 'docs/EVAL_CARRIER_OWNERSHIP.md'}
for p in old:
    if p.startswith('src/') and p not in {
        'src/eval.c', 'src/eval/carrier_values.c', 'src/eval/carrier_values.h',
        'src/eval/carrier_runtime.c', 'src/eval/carrier_runtime.h'}:
        assert old[p] == new[p], p
def provider(path):
    marker = '/.test_output/full-collector-source-20261123/'
    return rows['journal/' + path.split(marker, 1)[1]] if marker in path else rows['provider/' + path]

before = prior_record('source/src/eval.c')
addition = b'''                if (env_carrier_enabled(g_carrier_environment) && last_underscore > name + 5) {
                    const char *start = name + 5;
                    char *element = strndup(start, (size_t)(last_underscore - start));
                    if (!element) { fputs("I cannot retain the list constructor identity.\\n", stderr); exit(1); }
                    if (env_get_struct(env, element)) env_carrier_track_legacy_list(env, list);
                    free(element);
                }
'''
assert rows['source/src/eval.c'].count(addition) == 1
assert rows['source/src/eval.c'].replace(addition, b'') == before
for p, digest in js('journal/qualified-source').items():
    assert new[p] == digest, p
for name in ['local-before', 'local-before-san', 'explicit-free']:
    assert [r['exit'] for r in js('journal/' + name)] == [0, 1]
assert b'I cannot trace active interpreter roots.' in rows['journal/local-before-run.stderr']
assert b'heap-buffer-overflow' in rows['journal/local-before-san-run.stderr']
assert b'16-byte region' in rows['journal/local-before-san-run.stderr']
free = rows['journal/explicit-free-run.stderr']
assert b'heap-use-after-free' in free and b'valid_value' in free
assert b'list_int_free' in free and b'list_int_new' in free
assert b'(list_LowerRootItem_free items)' in rows['journal/explicit-free-probe.c']
for r in m['files']:
    if r['path'].startswith('journal/before/src/'):
        path = r['path'].removeprefix('journal/before/')
        assert r['sha256'] == old[path], path
repair = js('journal/repair')
assert len(repair['commands']) == 21 and all(r['exit'] == 0 for r in repair['commands'])
for p, digest in repair['providers'].items():
    assert h(provider(p)) == digest
    assert any(r['sha256'] == digest for r in prior['files']), p
for mode in ['normal', 'san', 'production']:
    for subject in ['eval', 'env', 'repl', 'carrier_runtime', 'carrier_values']:
        command = next(r for r in repair['commands'] if r['label'] == mode + '-compile-' + subject)
        assert '-Werror' in command['args']
        assert ('-O3' if mode == 'production' else '-O1') in command['args']
        if mode == 'san':
            assert '-fsanitize=address,undefined' in command['args']
for name, n in [('neighbors', 76), ('field-controls', 12)]:
    commands = js('journal/' + name)
    assert len(commands) == n and all(r['exit'] == 0 for r in commands)
setup = js('journal/field-root-setup/field-controls')
assert setup[-1]['exit'] == 1 and 'legacy_list_values-run' in setup[-1]['label']
controls = js('journal/final-controls')
assert len(controls) == 15
assert all(r['exit'] == (1 if r['label'] == 'san-default-admission' else 0) for r in controls)
for mode in ['normal', 'san', 'production']:
    for name, n in [('default34', 34), ('private41', 41), ('current-private41', 41)]:
        err = rows['journal/' + mode + '-' + name + '.stderr']
        assert ('Ran ' + str(n) + ' tests').encode() in err and b'\nOK\n' in err
        assert b'AddressSanitizer:' not in err and b'runtime error:' not in err
    assert b'Ran 7 tests' in rows['journal/' + mode + '-default-admission.stderr']
    make = next(r for r in controls if r['label'] == mode + '-final-make')['args']
    assert 'test-eval-legacy-record-lists' in make and 'test-carrier-legacy-list-values' in make
for mode in ['normal', 'san']:
    out = rows['journal/' + mode + '-eval-run.stdout']
    assert out.count(b'  Testing ') == 134 and b'All eval tests passed!' in out
    out = rows['journal/' + mode + '-checker-run.stdout']
    assert out.count(b'  Testing ') == 88 and b'All typechecker tests passed!' in out
    assert b'delta=0 calls=128.' in rows['journal/' + mode + '-native-bytes-run.stdout']
err = rows['journal/san-default-admission.stderr']
assert b'FAILED (failures=3)' in err
assert err.count(b'168 byte(s) leaked in 6 allocation(s)') == 1
assert err.count(b'172 byte(s) leaked in 6 allocation(s)') == 2

def events(name):
    result = []
    for line in rows['journal/' + name + '/stderr'].splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict): result.append(event)
        except ValueError:
            pass
    return result

for label, directory, expected in [('qualified', 'native', (1, 8, 7)),
                                  ('repaired-qualified', 'repaired-native', (1, 161, 160)),
                                  ('ordinary-qualified', 'ordinary-native', (0, 1306, 1306))]:
    q = js('journal/' + label)
    assert (q['exit'], q['starts'], q['completions']) == expected
    assert q['before'] == q['after'] and len(q['before']) == 1340
    assert q['timeout'] == 600 and q['shadow_timeout'] == 'default10' and not q['timed_out']
    # My first run is the old source. My repaired runs record modified bytes before commit.
    for p, digest in q['after'].items():
        assert digest == (old[p] if label == 'qualified' else new[p]), p
    assert h(rows['provider/' + q['args'][0]]) == q['image_sha256']
    for p, digest in q.get('providers', {}).items():
        assert h(provider(p)) == digest
    ev = events(directory)
    starts = [v for v in ev if v.get('event') == 'shadow-start']
    ends = [v for v in ev if v.get('event') == 'shadow-complete']
    assert len(starts) == expected[1] and len(ends) == expected[2]
    assert all(v.get('failures') == 0 for v in ends)
    assert [v for v in ev if v.get('event') == 'cseed-shadow-phase'] == q['phases']
    status = q['phases'][-1]
    assert status['reaped'] and status['deadline_seconds'] == 10
    if label == 'ordinary-qualified':
        assert status['completed'] and not status['parent_timeout'] and status['signal'] == status['exit_code'] == 0
        assert h(rows['journal/ordinary-native/seed']) == q['product']['sha256']
        assert js('journal/ordinary-seed-help')['exit'] == 0
        assert js('journal/ordinary-seed-help')['sha256'] == q['product']['sha256']
    else:
        assert q['product'] is None and not status['completed']
        observed = [v for v in ev if v.get('event') == 'collection-observation']
        assert observed == q['collections']
        if label == 'qualified':
            assert not status['parent_timeout'] and status['exit_code'] == 1
            assert observed[-1]['calls'] == 1461 and not observed[-1]['result']
        else:
            assert status['parent_timeout'] and status['signal'] == 14
            assert observed[-1]['calls'] == 65536 and observed[-1]['total_symbols'] == 654262134
            assert observed[-1]['collector_nanoseconds'] == 3765637591
for name, count in [('lexical', 10), ('record-payload', 7)]:
    q = js('journal/' + name + '-qualified')
    assert (q['tests'], q['failures'], q['errors'], q['success']) == (count, 0, 0, True)
    commands = js('journal/' + name + '-commands')
    expected = [1 if i in {2, 3, 4, 5, 6, 23} else 0 for i in range(24)] if name == 'lexical' else [0] * 12 + [1] * 3
    assert [c['exit'] for c in commands] == expected
    for command in commands:
        assert command['shadow_timeout'] is None
        assert command['timeout'] in ({10, 120} if name == 'lexical' else {10, 30})
        for item in command.get('inputs', []):
            assert h(rows['journal/' + name + '-inputs/' + item['sha256']]) == item['sha256']
assert js('journal/issue')['issue'] == 1231
assert js('journal/remaining-issues')['cost']['issue'] == 1232
assert js('journal/remaining-issues')['free']['issue'] == 1233
assert all(r['exit'] == 0 for r in js('journal/policy'))
assert js('journal/final-image-guards') == json.loads(rows['baseline/image-guards.json'])
closure = js('journal/product-closure')
assert len(closure) == 9
for p, digest in closure.items():
    assert h(provider(p)) == digest
assert m['acceptance'] == dict(
    legacy_constructor_roots=True, explicit_borrowed_layout=True, focused_legacy_lsan=True,
    ordinary_compiler_source=True, default_repl34=True, private_repl41=True,
    full_private_compiler_source=False, explicit_free_coordination=False,
    default_aggregate_ownership=False, fresh_shared_provider_build=False,
    paired_platform=False, raw_stage=False, whole_release=False)
print(f'PASS {len(rows)} records / {len(new)} exact Git inputs / {len(images)} native ELF images; '
      'ordinary 1306 complete, private layout prerequisite checked; explicit-free/cost/default/raw/paired/release open')
