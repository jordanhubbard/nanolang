#!/usr/bin/env python3
"""I verify accepted REPL source journals and their explicit remaining ownership failures."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess

base = Path(__file__).resolve().parent
m = json.loads((base / 'manifest.json').read_text())
h = lambda b: hashlib.sha256(b).hexdigest()
prior_bytes = (base / m['base_manifest']['path']).read_bytes()
assert h(prior_bytes) == m['base_manifest']['sha256']
prior = json.loads(prior_bytes)
rows = {}
images = set()
local_objects = set()
for r in m['files']:
    artifact = base / r['artifact']
    if artifact.parent.resolve() == (base / 'objects').resolve():
        local_objects.add(artifact.resolve())
    z = artifact.read_bytes()
    assert h(z) == r['compressed_sha256'], r['path']
    b = gzip.decompress(z)
    assert len(b) == r['bytes'] and h(b) == r['sha256'], r['path']
    assert r['path'] not in rows
    rows[r['path']] = b
    if b.startswith(b'\x7fELF') and int.from_bytes(b[16:18], 'little') in (2, 3):
        assert int.from_bytes(b[18:20], 'little') == 183
        images.add(h(b))
assert local_objects == {p.resolve() for p in (base / 'objects').glob('*.gz')}

proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=base,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def gitbytes(pin, path):
    proc.stdin.write((pin + ':' + path + '\n').encode())
    proc.stdin.flush()
    header = proc.stdout.readline().split()
    assert header[1] == b'blob'
    b = proc.stdout.read(int(header[2]))
    assert proc.stdout.read(1) == b'\n'
    return b

new = {}
for r in m['git_sources']:
    b = gitbytes(m['source_pin'], r['path'])
    assert h(b) == r['sha256'] and b == rows['source/' + r['path']]
    new[r['path']] = r['sha256']
old = {r['path']: r['sha256'] for r in prior['git_sources']}
allowed = {'Makefile.gnu', 'src/repl.c', 'docs/ROADMAP.md', 'docs/FEATURES.md',
           'docs/EVAL_CALLABLE_OWNERSHIP.md', 'docs/REPL_SESSIONS.md',
           'tests/test_repl_admission.py', 'tests/test_repl_session.py',
           'tests/support/repl_save_faults.c'}
assert {p for p, v in new.items() if old.get(p) != v} == allowed
for p in ['tests/test_repl_state.py', 'tests/test_repl_callable_lifetimes.py',
          'tests/native_process.py', 'src/repl_main.c', 'src/env.c', 'src/module.c',
          'src/eval.c', 'src/typechecker.c', 'src/nanolang.h', 'src/nanoisa/nvm2c.c']:
    assert old[p] == new[p], p

def js(name):
    return json.loads(rows['journal/' + name + '.json'])

def prior_record(name):
    r = next(r for r in prior['files'] if r['path'] == name)
    z = (base / m['base_manifest']['path']).parent.joinpath(r['artifact']).read_bytes()
    assert h(z) == r['compressed_sha256']
    b = gzip.decompress(z)
    assert len(b) == r['bytes'] and h(b) == r['sha256']
    return b

assert json.loads(prior_record('journal/loaded-save.json'))['saved_bytes'] == 0
assert b'=> 7' in rows['journal/failed-type-before.stdout']
assert b'Cannot determine struct type' in rows['journal/nominal-binding-before.stderr']
assert b'FAILED (failures=6)' in rows['journal/admission-baseline.stderr']
assert b'=> 11' in rows['journal/first-owner-before.stderr']
assert b'=> 71' in rows['journal/origin-before.stderr']
assert b'10 byte(s) leaked in 1 allocation(s)' in rows['journal/first-san-tests.stderr']
for p, digest in js('initial-session-source').items():
    if p == 'src/repl.c':
        assert h(rows['journal/initial-session-repl.c']) == digest
    elif p == 'tests/test_repl_session.py':
        assert h(rows['journal/initial-session-tests-14.py']) == digest
for p, digest in js('session-source').items():
    assert new[p] == digest, p
assert b'NANO_REPL_FAIL_SAVE' not in rows['source/src/repl.c']
assert b'NANO_REPL_FAIL_SAVE' in rows['source/tests/support/repl_save_faults.c']

qualified = js('session-qualified')
assert len(qualified['commands']) == 9 and all(r['exit'] == 0 for r in qualified['commands'])
for path, digest in qualified['providers'].items():
    assert h(rows['provider/' + path]) == digest
    assert rows['provider/' + path] == prior_record('provider/' + path)
original = {r['mode']: r['link'] for r in js('original-links')}
for profile in ['normal', 'san', 'production']:
    compile_row = next(r for r in qualified['commands'] if r['label'] == profile + '-session-compile')
    link_row = next(r for r in qualified['commands'] if r['label'] == profile + '-session-link')
    old_link = original[profile]
    assert compile_row['args'][:-4] == old_link[:old_link.index('-o')]
    assert compile_row['args'][-4:-2] == ['-c', 'src/repl.c']
    normalized = list(link_row['args'])
    normalized[normalized.index('-o') + 1] = old_link[old_link.index('-o') + 1]
    normalized[normalized.index(compile_row['args'][-1])] = next(p for p in old_link if p.endswith('/repl.o'))
    assert normalized == old_link
    if profile == 'production':
        assert '-O3' in compile_row['args'] and '-ftree-vectorize' in compile_row['args'] and '-Werror' in compile_row['args']
    if profile == 'san':
        assert '-fsanitize=address,undefined' in compile_row['args']
    text = rows['journal/' + profile + '-session-tests.stderr']
    assert b'Ran 3 tests' in text and b'Ran 15 tests' in text and b'Ran 16 tests' in text
    assert b'FAILED' not in text

# I decode successful saves independently of the C parser and test encoder.
magic = b'# nanolang-repl-session v1\n'
def decode(data):
    assert data.startswith(magic)
    cursor, records = len(magic), []
    while cursor < len(data):
        end = data.index(b'\n', cursor)
        fields = data[cursor:end].split(b' ')
        assert len(fields) == 3 and fields[0] in [b'input', b'load', b'reload', b'clear']
        assert fields[1].isdigit() and fields[2].isdigit()
        origin, size = map(int, fields[1:]); cursor = end + 1
        assert origin <= len(data) - cursor
        path = data[cursor:cursor + origin]; cursor += origin
        assert size <= len(data) - cursor
        source = data[cursor:cursor + size]; cursor += size
        assert cursor < len(data) and data[cursor] == 10
        cursor += 1
        assert b'\0' not in path + source
        if fields[0] == b'clear':
            assert origin == size == 0
        else:
            assert path.startswith(b'/') and source and not source.startswith(b'# nanolang-repl-session ')
        records.append((fields[0], path, source))
    assert len(records) <= 4096
    return records

import re
save_count = fault_count = 0
for profile in ['normal', 'san', 'production']:
    q = js(profile + '-session-qualified')
    assert (q['tests'], q['failures'], q['errors'], q['success']) == (16, 0, 0, True)
    assert q['source_sha256'] == new['tests/test_repl_session.py'] and not q['selection']
    commands = js(profile + '-session-commands')
    assert len(commands) == 36
    saves = faults = 0
    for index, r in enumerate(commands):
        assert r['exit'] == 0 and r['timeout'] == 30
        assert r['environment']['ASAN_OPTIONS'] == 'detect_leaks=1'
        assert h(rows['provider/' + r['args'][0]]) == r['image_sha256']
        capture = 'journal/' + profile + '-session-inputs/'
        assert h(rows[capture + r['stdin_sha256']]) == r['stdin_sha256']
        for inp in r['before'] + r['after']:
            assert h(rows[capture + inp['sha256']]) == inp['sha256']
        before = {p['path']: p['sha256'] for p in r['before']}
        after = {p['path']: p['sha256'] for p in r['after']}
        out = rows['journal/' + profile + '-session-' + str(index) + '.stdout']
        err = rows['journal/' + profile + '-session-' + str(index) + '.stderr']
        assert b'AddressSanitizer:' not in err and b'runtime error:' not in err
        for count, path in re.findall(rb"; saved (\d+) fragment\(s\) to '([^']*)'", out):
            data = rows[capture + after[path.decode()]]
            assert len(decode(data)) == int(count)
            saves += 1
        if 'NANO_REPL_FAIL_SAVE' in r['environment']:
            target = r['environment']['NANO_REPL_FAIL_SAVE_TARGET']
            assert r['environment']['NANO_REPL_FAIL_SAVE'] in ['write', 'close']
            assert rows[capture + before[target]] == b'prior content'
            assert before[target] == after[target]
            assert not any(p.startswith(target + '.nano-repl-') for p in after)
            assert b'I cannot save' in err and b'; saved ' not in out
            faults += 1
    assert saves == 13 and faults == 2
    save_count += saves; fault_count += faults

observed = js('observed')
assert len(observed) == 8
for r in observed:
    assert r['exit'] == (1 if r['label'] == 'san-admission' else 0)
for label, count, failures in [('normal-admission', 7, 0), ('san-admission', 7, 3),
                               ('production-admission', 7, 0), ('san-admission-broad', 7, 0),
                               ('san-admission-focused', 4, 0)]:
    q = js(label + '-qualified')
    assert (q['tests'], q['failures'], q['errors']) == (count, failures, 0)
    assert q['source_sha256'] == new['tests/test_repl_admission.py']
    for r in js(label + '-commands'):
        assert r['timeout'] == 30
        assert r['environment']['ASAN_OPTIONS'] == ('detect_leaks=0' if label == 'san-admission-broad' else 'detect_leaks=1')
        capture = 'journal/' + label + '-inputs/'
        assert h(rows[capture + r['stdin_sha256']]) == r['stdin_sha256']
        for inp in r['before'] + r['after']:
            assert h(rows[capture + inp['sha256']]) == inp['sha256']
reports = [rows['journal/san-admission-' + str(i) + '.stderr'] for i in range(7)]
leaks = sorted((int(a), int(b)) for text in reports for a, b in re.findall(rb'SUMMARY: AddressSanitizer: (\d+) byte\(s\) leaked in (\d+) allocation\(s\)', text))
assert leaks == [(168, 6), (254, 12), (254, 12)]
assert all(b'heap-use-after-free' not in text and b'runtime error:' not in text for text in reports)
assert len(js('san-admission-focused-qualified')['selection']) == 4
assert all(r['exit'] == 0 for r in js('policy'))
assert b'0 reserved names' in rows['journal/policy.stdout']
assert js('final-image-guards') == json.loads(rows['baseline/image-guards.json'])
for path, digest in js('product-closure').items():
    assert h(rows['provider/' + path]) == digest
assert m['acceptance'] == dict(source_session_replay=True, repl_admission_functional=True,
    nominal_binding_names=True, strict_repl_controls=True, shared_providers_reused_unchanged=True,
    fresh_shared_provider_build=False, complete_aggregate_lsan=False, standalone_compiler_input=False,
    complete_repl_ownership=False, default_string_activation=False, paired_platform=False,
    raw_stage=False, whole_release=False)
proc.stdin.close()
assert proc.wait() == 0
print(f'PASS {len(rows)} records / {len(new)} exact Git inputs / {len(images)} native ELF images; '
      f'34 strict REPL methods/profile, {save_count} decoded saves, {fault_count} preserved I/O faults; '
      'three complete aggregate LSan failures retained; full ownership/platform/raw/release open')
