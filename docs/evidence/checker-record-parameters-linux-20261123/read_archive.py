#!/usr/bin/env python3
"""I verify checker-only record slots and borrowed nested-field identities."""
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
    (base / r['artifact']).resolve() for r in m['files']
    if r['artifact'].startswith('objects/')}

def prior_record(name):
    r = next(r for r in prior['files'] if r['path'] == name)
    z = (priorpath.parent / r['artifact']).read_bytes()
    assert h(z) == r['compressed_sha256']
    b = gzip.decompress(z)
    assert h(b) == r['sha256'] and len(b) == r['bytes']
    return b

def js(name):
    return json.loads(rows['journal/' + name + '.json'])

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
    'src/typechecker.c', 'Makefile.gnu', 'tests/test_checker_record_parameters.c',
    'docs/ROADMAP.md', 'docs/EVAL_CALLABLE_OWNERSHIP.md'}
for p in old:
    if p.startswith('src/') and p != 'src/typechecker.c':
        assert old[p] == new[p], p
checker = prior_record('source/src/typechecker.c')
assert rows['journal/typechecker-before.c'] == checker
placeholder = checker.replace(
    b'val = create_struct(item->as.function.params[j].struct_type_name, NULL, NULL, 0);',
    b'/* I retain nominal metadata below, as in type_check.\n'
    b'                     * A checker-only slot does not need a runtime record. */\n'
    b'                    val = create_void();')
assert placeholder != checker
assert rows['journal/placeholder-only/typechecker.c'] == placeholder
final = placeholder.replace(
    b'/* Return the struct type name for this field */\n'
    b'                        return strdup(sdef->field_type_names[i]);',
    b'/* I borrow this declaration identity, like other queries. */\n'
    b'                        return sdef->field_type_names[i];')
assert final != placeholder and final == rows['source/src/typechecker.c']
assert [r['exit'] for r in js('baseline')] == [0, 0, 0, 1]
assert b'615 byte(s) leaked in 51 allocation(s)' in rows['journal/san-baseline-run.stderr']
assert b'36 byte(s) leaked in 6 allocation(s)' in rows['journal/placeholder-only/san-focused-run.stderr']
assert b'obj/repl_save_faults.so' in rows['journal/first-qualification/normal-repl-strict.stderr']
assert b'FAILED (failures=1)' in rows['journal/first-qualification/normal-repl-strict.stderr']
assert b'.test_output/src/runtime/list_int.c' in rows['journal/misplaced-compiler/lexical-tests.stdout']
location = js('compiler-location')
assert h(rows['journal/compiler']) == location['sha256']
assert h(rows['provider/' + location['actual']]) == location['sha256']

q = js('qualified')
commands = {r['label']: r for r in q['commands']}
assert len(commands) == len(q['commands']) == 25
for label, r in commands.items():
    assert r['exit'] == (1 if label == 'san-admission-complete' else 0), label
    if r['exit'] == 0 and label.endswith(('-run', '-strict', '-functional', '-complete')):
        assert b'AddressSanitizer:' not in rows['journal/' + label + '.stderr']
        assert b'runtime error:' not in rows['journal/' + label + '.stderr']
for profile in ['normal', 'san', 'production']:
    assert b'I pass 36 repeated checker record-parameter controls.' in rows['journal/' + profile + '-focused-run.stdout']
    assert rows['journal/' + profile + '-original-checker-run.stdout'].count(b'  Testing ') == 88
    assert b'Ran 34 tests' in rows['journal/' + profile + '-repl-strict.stderr']
    assert b'\nOK\n' in rows['journal/' + profile + '-repl-strict.stderr']
    assert commands[profile + '-repl-strict']['detect_leaks']
    assert b'Ran 7 tests' in rows['journal/' + profile + '-admission-complete.stderr']
assert not commands['san-original-checker-run']['detect_leaks']
assert not commands['san-admission-functional']['detect_leaks']
assert b'FAILED (failures=3)' in rows['journal/san-admission-complete.stderr']
assert rows['journal/san-admission-complete.stderr'].count(b'168 byte(s) leaked in 6 allocation(s)') == 1
assert rows['journal/san-admission-complete.stderr'].count(b'172 byte(s) leaked in 6 allocation(s)') == 2
plans = json.loads(prior_record('journal/original-links.json'))
for plan in plans:
    profile = plan['mode']
    original = plan['link']
    flags = original[:original.index('-o')]
    assert commands[profile + '-compile']['args'][:-4] == flags
    assert commands[profile + '-compile']['args'][-4:-1] == ['-c', 'src/typechecker.c', '-o']
    assert '-Werror' in flags
    if profile == 'production':
        assert '-O3' in flags
    for r in q['commands']:
        if r['label'].startswith(profile + '-') and r['label'].endswith('-link'):
            actual = r['args']
            for p in original:
                if p.endswith(('.o', '.a')) and not p.endswith(('/repl.o', '/repl_main.o', '/typechecker.o')):
                    assert p in actual
for path, digest in q['providers'].items():
    assert h(rows['provider/' + path]) == digest
    assert any(r['sha256'] == digest for r in prior['files']), path
for p, digest in js('source').items():
    assert new[p] == digest
for name, n in [('lexical', 10), ('record-payload', 7)]:
    r = js(name + '-qualified')
    assert (r['tests'], r['failures'], r['errors'], r['success']) == (n, 0, 0, True)
    for command in js(name + '-commands'):
        for inp in command.get('inputs', []):
            if inp:
                assert h(rows['journal/' + name + '-inputs/' + inp['sha256']]) == inp['sha256']
        if command.get('product'):
            product = command['product']
            assert h(rows['journal/' + name + '-inputs/' + product['sha256']]) == product['sha256']
s = js('full-source-checker-record-qualified')
assert s['exit'] == 1 and not s['timed_out'] and s['timeout'] == 600
assert s['shadow_timeout'] == 'default' and s['before'] == s['after']
assert all(new[p] == digest for p, digest in s['after'].items())
assert s['compiler_sha256'] == location['sha256']
assert 'product' not in s and 'journal/full-source-checker-record/seed' not in rows
assert b'I stopped shadow execution after 10 seconds.' in rows['journal/full-source-checker-record.stderr']
events = [json.loads(line) for line in rows['journal/full-source-checker-record.stderr'].decode().splitlines()
          if line.startswith('{"event":"shadow-')]
starts = [r for r in events if r['event'] == 'shadow-start']
ends = [r for r in events if r['event'] == 'shadow-complete']
assert len(starts) == len(ends) == 1306 and all(r['failures'] == 0 for r in ends)
assert [(r['source'], r['name'], r['line']) for r in starts] == [(r['source'], r['name'], r['line']) for r in ends]
assert 'journal/full-source-checker-record-help.json' not in rows
assert js('shadow-followup')['issue'] > 1228
assert len(js('make-controls')) == 3 and all(r['exit'] == 0 for r in js('make-controls'))
for profile in ['normal', 'san', 'production']:
    assert b'I pass 36 repeated checker record-parameter controls.' in rows['journal/' + profile + '-make.stdout']
assert all(r['exit'] == 0 for r in js('policy'))
assert js('final-image-guards') == json.loads(rows['baseline/image-guards.json'])
for path, digest in js('product-closure').items():
    assert h(rows['provider/' + path]) == digest
assert m['acceptance'] == dict(
    checker_parameter_metadata=True, nested_field_borrow=True, strict_focused_lsan=True,
    strict_repl_controls=True, shared_providers_reused_unchanged=True,
    fresh_shared_provider_build=False, full_compiler_source=False, complete_aggregate_lsan=False,
    complete_repl_ownership=False, default_string_activation=False,
    paired_platform=False, raw_stage=False, whole_release=False)
print(f'PASS {len(rows)} records / {len(new)} exact Git inputs / {len(images)} native ELF images; '
      '36 strict checker controls/profile, original checker88 and REPL34/profile, '
      '1306 shadow completions followed by supervisor refusal; '
      'all three runtime-copy LSan failures retained; full source/release open')
