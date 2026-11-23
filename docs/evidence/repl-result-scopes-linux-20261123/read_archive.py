#!/usr/bin/env python3
"""I verify displayed-result storage and distinct compiler shadow terminals."""
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
    'src/repl.c', 'Makefile.gnu', 'tests/test_repl_result_scopes.c',
    'docs/ROADMAP.md', 'docs/EVAL_CALLABLE_OWNERSHIP.md', 'docs/EVAL_CARRIER_OWNERSHIP.md'}
for p in old:
    if p.startswith('src/') and p != 'src/repl.c':
        assert old[p] == new[p], p
repl = prior_record('source/src/repl.c')
assert rows['result/repl-before.c'] == repl
fixed = repl.replace(b'#include "interpreter_ffi.h"',
                     b'#include "interpreter_ffi.h"\n#include "eval/carrier_runtime.h"')
fixed = fixed.replace(b'        Value result = repl_eval_node(node, env);',
    b'        /* I keep owned results in the caller until display has finished. */\n'
    b'        env_carrier_enter(env);\n        Value result = repl_eval_node(node, env);')
fixed = fixed.replace(b'        show_result(result); retire_callable_result(result);',
    b'        show_result(result); retire_callable_result(result);\n        env_carrier_leave(env);')
assert fixed != repl and fixed == rows['source/src/repl.c']
assert rows['result/test-before.c'] == rows['source/tests/test_repl_result_scopes.c']
assert [r['exit'] for r in js('result/baseline')] == [0, 1]
positive = b'I observe and retire my 2MiB positive control: 2097193 bytes.'
passed = b'I retire all 512 displayed STRING results and preserve my borrowed binding.'
assert positive in rows['result/before-run.stdout']
assert b'I retain 103 bytes/2 objects; I require 49/1.' in rows['result/before-run.stderr']
assert js('result/issue')['issue'] == 1230
assert b'for (int round = 0; round < 128; ++round)' in rows['source/tests/test_repl_result_scopes.c']
assert b'fresh == 128 && joined == 128 && returned == 128 && borrowed == 128' in rows['source/tests/test_repl_result_scopes.c']
for journal in ['result', 'private-before']:
    q = js(journal + '/qualified')
    assert len(q['commands']) == (21 if journal == 'result' else 18)
    assert all(r['exit'] == 0 for r in q['commands'])
    for path, digest in q['providers'].items():
        assert h(rows['provider/' + path]) == digest
        assert any(r['sha256'] == digest for r in prior['files']), path
    for p, digest in js(journal + '/source').items():
        assert digest == (old[p] if journal == 'private-before' and p == 'src/repl.c' else new[p]), p
commands = {r['label']: r for r in js('result/qualified')['commands']}
for mode in ['normal', 'san', 'production']:
    assert positive in rows['result/' + mode + '-result-run.stdout']
    assert passed in rows['result/' + mode + '-result-run.stdout']
    assert b'AddressSanitizer:' not in rows['result/' + mode + '-result-run.stderr']
    for suffix, n in [('default34', 34), ('private41', 41)]:
        err = rows['result/' + mode + '-' + suffix + '.stderr']
        assert ('Ran ' + str(n) + ' tests').encode() in err and b'\nOK\n' in err
        assert b'AddressSanitizer:' not in err and b'runtime error:' not in err
        assert commands[mode + '-' + suffix]['environment']['ASAN_OPTIONS'] == 'detect_leaks=1'
    default = commands[mode + '-repl-link']['args']
    private = commands[mode + '-private-link']['args']
    normalized = list(private[:-1])
    assert private[-1].endswith('/' + mode + '-environment-shim.o')
    env_index = next(i for i, p in enumerate(normalized) if p.endswith('/' + mode + '-environment-original.o'))
    normalized[env_index] = next(p for p in default if p.endswith('/env.o'))
    normalized[normalized.index('-o') + 1] = default[default.index('-o') + 1]
    assert normalized == default
    flags = commands[mode + '-repl-compile']['args']
    assert '-Werror' in flags and ('-O3' in flags if mode == 'production' else '-O1' in flags)
    if mode == 'san':
        assert '-fsanitize=address,undefined' in flags

controls = js('result/final-controls')
assert len(controls) == 12
assert all(r['exit'] == (1 if r['label'] == 'san-default-admission' else 0) for r in controls)
for mode in ['normal', 'san', 'production']:
    for suffix in ['strings-run', 'make']:
        assert positive in rows['result/' + mode + '-' + suffix + '.stdout']
        assert passed in rows['result/' + mode + '-' + suffix + '.stdout']
        assert b'AddressSanitizer:' not in rows['result/' + mode + '-' + suffix + '.stderr']
    assert b'Ran 7 tests' in rows['result/' + mode + '-default-admission.stderr']
err = rows['result/san-default-admission.stderr']
assert b'FAILED (failures=3)' in err
assert err.count(b'168 byte(s) leaked in 6 allocation(s)') == 1
assert err.count(b'172 byte(s) leaked in 6 allocation(s)') == 2
# I verify the generated selector without assuming this reader's checkout path.
actual_strings = rows['result/strings-only.c'].decode()
line = next(line for line in actual_strings.splitlines() if line.startswith('#include "') and line.endswith('/src/repl.c"'))
strings = rows['source/tests/test_repl_result_scopes.c'].decode().replace('#include "../src/repl.c"', line)
strings = strings.replace('    REQUIRE(env_carrier_enable(env));\n    REQUIRE(env_carrier_enable_collection(env, NULL, 0));',
                          '    REQUIRE(env_carrier_enable_strings(env));')
assert actual_strings == strings
observed_calls = 0
for mode in ['normal', 'san', 'production']:
    assert b'Ran 18 tests' in rows['private-before/' + mode + '-original18.stderr']
    assert b'\nOK\n' in rows['private-before/' + mode + '-original18.stderr']
    for subject, n in [('session', 16), ('admission', 7)]:
        label = mode + '-' + subject
        q = js('private-before/' + label + '-qualified')
        assert (q['tests'], q['failures'], q['errors'], q['success']) == (n, 0, 0, True)
        assert q['selection'] == [] and q['source_sha256'] == new[q['source']]
        for call in js('private-before/' + label + '-commands'):
            assert call['timeout'] == 30 and call['exit'] == 0
            assert call['environment']['ASAN_OPTIONS'] == 'detect_leaks=1'
            assert h(rows['private-before/' + label + '-inputs/' + call['stdin_sha256']]) == call['stdin_sha256']
            assert h(rows['provider/' + call['args'][0]]) == call['image_sha256']
            for snapshot in call['before'] + call['after']:
                if snapshot:
                    assert h(rows['private-before/' + label + '-inputs/' + snapshot['sha256']]) == snapshot['sha256']
            observed_calls += 1
assert observed_calls == 129

original = json.loads(prior_record('journal/full-source-checker-record-qualified.json'))
assert original['exit'] == 1 and original['timeout'] == 600 and not original['timed_out']
assert b'I stopped shadow execution after 10 seconds.' in prior_record('journal/full-source-checker-record.stderr')
traced = js('diagnostic/qualified')
phase = js('diagnostic/phase-only/qualified')
assert (traced['exit'], traced['starts'], traced['completions'], traced['assertion_failures']) == (1, 812, 811, 0)
assert traced['product'] is None and not traced['timed_out']
status = traced['phases'][-1]
assert status['phase'] == 'supervisor-status' and status['reaped'] and status['signal'] == 14
assert status['parent_timeout'] and not status['completed'] and status['deadline_seconds'] == 10
trace = rows['diagnostic/native/process.strace']
assert b'+++ killed by SIGALRM +++' in trace and b'kill(' in trace
assert b'si_utime=409' in trace and b'si_stime=245' in trace
assert (phase['exit'], phase['starts'], phase['completions'], phase['assertion_failures']) == (0, 1306, 1306, 0)
assert not phase['timed_out'] and phase['seconds'] < 600
assert [p['phase'] for p in phase['phases']] == [
    'dispatch-start', 'report-start', 'report-returned', 'dispatch-returned',
    'completion-written', 'flush-start', 'flush-end', 'supervisor-status']
status = phase['phases'][-1]
assert status['reaped'] and status['completed'] and status['exit_code'] == status['signal'] == 0
assert not status['parent_timeout'] and not status['clock_failure'] and status['deadline_seconds'] == 10
assert status['child_pid'] == phase['phases'][0]['pid'] and status['pid'] == phase['pid']
clock = lambda p: p['monotonic_seconds'] * 10**9 + p['monotonic_nanoseconds']
interval = clock(status) - (status['parent_start_seconds'] * 10**9 + status['parent_start_nanoseconds'])
gap = clock(status) - clock(phase['phases'][-2])
assert interval == 9386556277 and gap == 428221727 and interval < 10**10
for q in [traced, phase]:
    assert q['compiler_sha256'] == original['compiler_sha256']
    assert q['before'] == q['after'] and len(q['before']) == 1340
    assert q['timeout'] == 600 and q['shadow_timeout'] == 'default10'
    for p, digest in q['after'].items():
        assert digest == (old[p] if p == 'src/repl.c' else new[p]), p
    assert q['environment']['NANO_SHADOW_TRACE'] == q['environment']['NANO_SHADOW_PHASE_TRACE'] == '1'
    assert h(rows['provider/' + q['args'][-4]]) == q['compiler_sha256']
assert h(rows['diagnostic/phase-only/native/seed']) == phase['product']['sha256']
assert js('diagnostic/phase-only/seed-help')['exit'] == 0
assert js('diagnostic/phase-only/seed-help')['sha256'] == phase['product']['sha256']
assert all(r['exit'] == 0 for r in js('result/policy'))
assert js('result/final-image-guards') == json.loads(rows['baseline/image-guards.json'])
for path, digest in js('result/product-closure').items():
    assert h(rows['provider/' + path]) == digest
assert m['acceptance'] == dict(
    private_repl41=True, display_gc_storage=True, default_repl34=True,
    default_complete_aggregate_lsan=False, phase_observed_shared_compiler_stage=True,
    historical_timeout_cause=False, ordinary_session_activation=False,
    fresh_shared_provider_build=False, total_native_heap=False, paired_platform=False,
    raw_stage=False, whole_release=False)
print(f'PASS {len(rows)} records / {len(new)} exact Git inputs / {len(images)} native ELF images; '
      'default34/private41 across three profiles, nine 512-result GC controls, '
      'distinct 811/1306 shadow terminals preserved; default/raw/paired/release open')
