#!/usr/bin/env python3
"""I verify my bounded REPL repair and retain its unsuccessful controls."""
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
allowed = {'Makefile.gnu', 'docs/ROADMAP.md', 'src/env.c', 'src/eval.c',
           'src/module.c', 'src/nanolang.h', 'src/repl.c', 'src/typechecker.c',
           'docs/EVAL_CALLABLE_OWNERSHIP.md', 'docs/EVAL_CARRIER_OWNERSHIP.md', 'docs/FEATURES.md',
           'tests/test_eval_tuple_vector_guards.c', 'tests/test_repl_state.py',
           'tests/test_eval_carrier_native_bytes.c'}
# My base receipt predates the already-published native-counter portability note.
assert {p for p, v in new.items() if old.get(p) != v} == allowed
for p in ['src/nanoisa/nvm2c.c', 'src/eval/carrier_runtime.c',
          'src/eval/carrier_heap.c', 'tests/test_eval.c', 'tests/test_typechecker.c',
          'tests/test_cseed_result_list_payloads.py']:
    assert old[p] == new[p], p
assert rows['source/tests/test_repl_callable_lifetimes.py'] == gitbytes(
    '7379f5514ffb74195afde300d31e7af8cd176dbc', 'tests/test_repl_callable_lifetimes.py')

def js(name):
    return json.loads(rows['journal/' + name + '.json'])

assert js('original')['exit'] != 0
assert b'parent' in rows['journal/original.stderr']
assert js('production')['exit'] == 2
assert b'alloc-size-larger-than' in rows['journal/production.stderr']
assert js('production-corrected')['exit'] == 0
assert js('production')['args'] == js('production-corrected')['args']
assert not any(a.startswith(('CFLAGS=', 'LDFLAGS='))
               for a in js('production-corrected')['args'])
production = rows['journal/production-corrected.stderr']
assert b'Ran 3 tests' in production and b'Ran 15 tests' in production
assert b'heap-buffer-overflow' in rows['journal/unfinished-original.stderr']
assert b'heap-use-after-free' in rows['journal/module-failure-run.stderr']
assert b'badly-typed' in rows['journal/unchecked-statement-before.stdout']
assert js('save-before')['exit'] == 0 and js('save-before')['saved_bytes'] == 0
assert js('san-state-qualified')['failures'] == 1
assert b'8 byte(s)' in rows['journal/san-state-3.stderr']

for name, count in [('normal-final-state-qualified', 15),
                    ('san-final-state-qualified', 15)]:
    q = js(name)
    assert (q['tests'], q['failures'], q['errors'], q['success']) == (count, 0, 0, True)
    assert q['source_sha256'] == new['tests/test_repl_state.py']
for profile in ['normal', 'san']:
    prefix = profile + '-final-state'
    commands = js(prefix + '-commands')
    assert len(commands) == 16
    for r in commands:
        assert r['exit'] == 0 and r['timeout'] == 30
        assert h(rows['journal/' + prefix + '-inputs/' + r['stdin_sha256']]) == r['stdin_sha256']
        for inp in r['inputs']:
            assert h(rows['journal/' + prefix + '-inputs/' + inp['sha256']]) == inp['sha256']
    text = rows['journal/' + profile + '-qualified.stdout']
    assert b'I verify eight deliberate tuple refusals' in text
    assert b'delta=0 calls=128' in text
    assert b'65536 linked allocations' in text
    err = rows['journal/' + profile + '-qualified.stderr']
    assert b'Ran 3 tests' in err and b'Ran 15 tests' in err
    assert b'AddressSanitizer:' not in err and b'runtime error:' not in err
    assert rows['journal/' + profile + '-eval-run-compacted.stdout'].count(b'  Testing ') == 134
    assert rows['journal/' + profile + '-checker-run-compacted.stdout'].count(b'  Testing ') == 88
assert len(js('neighbors-compacted')) == 8
assert all(r['exit'] == 0 for r in js('neighbors-compacted'))
assert all(not r['detect_leaks'] for r in js('neighbors-compacted')
           if r['label'].startswith('san-') and r['label'].endswith('-run'))
assert [r['exit'] for r in js('qualified')['commands']] == [0, 0]
assert b"WEXITSTATUS(status),1" in rows['source/tests/test_eval_tuple_vector_guards.c']
assert b"ASSERT_EQ(seen,'R')" in rows['source/tests/test_eval_tuple_vector_guards.c']
assert b'atexit(confirm_refusal)' in rows['source/tests/test_eval_tuple_vector_guards.c']
for path, digest in js('qualified-source').items():
    assert new[path] == digest, path
assert all(r['exit'] == 0 for r in js('built-compilers')['commands'])
for mode in ['baseline', 'strings']:
    q = js(mode + '-native-qualified')
    assert (q['tests'], q['failures'], q['errors'], q['success']) == (7, 0, 0, True)
    assert q['test_source_sha256'] == new['tests/test_cseed_result_list_payloads.py']
    for r in js(mode + '-native-commands'):
        for inp in r.get('inputs', []):
            if inp:
                assert h(rows['journal/' + mode + '-native-inputs/' + inp['sha256']]) == inp['sha256']
        if r.get('product'):
            p = r['product']
            assert h(rows['journal/' + mode + '-native-inputs/' + p['sha256']]) == p['sha256']
    q = js('full-source-' + mode + '-qualified')
    assert q['exit'] == 0 and not q['timed_out'] and q['timeout'] == 600
    assert q['shadow_timeout'] == 'default' and q['before'] == q['after']
    assert all(new[p] == v for p, v in q['after'].items())
    assert h(rows['provider/' + q['args'][0]]) == q['compiler_sha256']
    assert h(rows['journal/full-source-' + mode + '/seed']) == q['product']['sha256']
    events = [json.loads(s) for s in rows['journal/full-source-' + mode + '.stderr'].decode().splitlines()
              if s.startswith('{"event":"shadow-')]
    start = [e for e in events if e['event'] == 'shadow-start']
    end = [e for e in events if e['event'] == 'shadow-complete']
    assert len(start) == len(end) == 1306 and all(e['failures'] == 0 for e in end)
    assert [(e['source'], e['name'], e['line']) for e in start] == [
        (e['source'], e['name'], e['line']) for e in end]
    assert js('full-source-' + mode + '-help')['exit'] == 0
assert js('module-introspection')['compile_exit'] == js('module-introspection')['run_exit'] == 0
assert js('policy')['exit'] == 0
assert js('final-image-guards') == json.loads(rows['baseline/image-guards.json'])
for path, digest in js('product-closure').items():
    assert h(rows['provider/' + path]) == digest
assert m['acceptance'] == dict(strict_repl_build=True, positive_repl_state=True,
    tuple_vector_refusal=True, module_context_retirement=True, first_compiler_source_stage=True,
    rejected_module_lifetime=False, loaded_fragment_save=False, complete_repl=False,
    default_string_activation=False, paired_platform=False, raw_stage=False, whole_release=False)
proc.stdin.close()
assert proc.wait() == 0
print(f'PASS {len(rows)} records / {len(new)} exact Git inputs / {len(images)} native ELF images; '
      'REPL 18 methods, evaluator134/checker88, two1306-shadow source stages; '
      'rejected-module/save/default/platform/raw/release remain open')
