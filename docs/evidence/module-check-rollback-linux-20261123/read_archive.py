#!/usr/bin/env python3
"""I verify module declaration rollback, copied enum owners and borrowed API names."""
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
allowed = {'Makefile.gnu', 'docs/EVAL_CALLABLE_OWNERSHIP.md', 'docs/FEATURES.md',
           'docs/ROADMAP.md', 'src/env.c', 'src/module.c', 'src/nanolang.h',
           'src/typechecker.c', 'tests/test_module_rejection_rollback.c',
           'tests/test_env_scoping.c', 'tests/test_nominal_owner_context.c',
           'tests/test_eval_enum_ownership.c', 'tests/test_lexical_scope_boundaries.py'}
assert {p for p, v in new.items() if old.get(p) != v} == allowed
for p in ['src/repl.c', 'src/eval.c', 'src/nanoisa/nvm2c.c',
          'tests/test_repl_callable_lifetimes.py', 'tests/test_repl_state.py',
          'tests/test_typechecker.c', 'tests/test_eval.c']:
    assert old[p] == new[p], p
for p in ['tests/test_env_scoping.c', 'tests/test_nominal_owner_context.c',
          'tests/test_eval_enum_ownership.c', 'tests/test_lexical_scope_boundaries.py']:
    assert rows['source/' + p] == gitbytes(prior['source_pin'], p)

def js(name):
    return json.loads(rows['journal/' + name + '.json'])

def prior_record(name):
    r = next(r for r in prior['files'] if r['path'] == name)
    z = (base / m['base_manifest']['path']).parent.joinpath(r['artifact']).read_bytes()
    assert h(z) == r['compressed_sha256']
    b = gzip.decompress(z)
    assert h(b) == r['sha256'] and len(b) == r['bytes']
    return b

assert b'heap-use-after-free' in prior_record('journal/module-failure-run.stderr')
assert [r['exit'] for r in js('baseline')] == [0, -6]
assert b'!env_get_function_owned(env, "broken", "BrokenOwner")' in rows['journal/baseline-run.stderr']
assert [r['exit'] for r in js('first-qualified')['commands']] == [0, 2]
assert b'534 byte(s) leaked in 68 allocation(s)' in rows['journal/first-san-qualified.stderr']
assert [r['exit'] for r in js('direct-free-qualified')['commands']] == [0, 0]
assert [r['exit'] for r in js('borrowed-original')] == [0, -6]
assert b'free(): invalid pointer' in rows['journal/borrowed-run.stderr']
assert b'free(env->enums[i].module_name)' in rows['journal/direct-free-env.c']
assert b'free(env->enums[i].module_name)' not in rows['source/src/env.c']
checker = rows['source/src/typechecker.c']
assert checker.count(b'edef.variant_values = malloc(sizeof(int) * edef.variant_count);') == 2
assert checker.count(b'edef.module_name = env->current_module ? env_own_checker_allocation(env, strdup(env->current_module)) : NULL;') == 2
assert [r['exit'] for r in js('qualified')['commands']] == [0, 0]
for path, digest in js('qualified-source').items():
    if path == 'tests/test_module_rejection_rollback.c':
        assert h(rows['journal/before-helper-control.c']) == digest
    else:
        assert new[path] == digest, path
assert js('helper-source')['tests/test_module_rejection_rollback.c'] == new['tests/test_module_rejection_rollback.c']
assert len(js('helper-controls')) == 6 and all(r['exit'] == 0 for r in js('helper-controls'))
for profile in ['normal', 'san', 'production']:
    assert b'I exercise actual record-list generic/helper registration' in rows['journal/' + profile + '-helper-run.stdout']
    assert b'I withdraw all failed declaration kinds across 32 attempts' in rows['journal/' + profile + '-helper-run.stdout']
assert b'env_module_rollback(env, checkpoint);' in rows['source/src/module.c']
assert b'env_module_commit(checkpoint);' in rows['source/src/module.c']
unit = rows['source/tests/test_module_rejection_rollback.c']
assert b'iteration < 32' in unit and b'flag->variant_values[1] == -2' in unit
assert b'flag->variant_values[1] == 1' in unit
labels = [b'I withdraw all failed declaration kinds across 32 attempts',
          b'I withdraw failed namespace/function/enum/effect/global aliases',
          b'I preserve circular-import refusal',
          b'I recover the same source path after missing-file, parser and lexer failures',
          b'I retire rejected module metadata and all retained successful owners']
for profile in ['normal', 'san']:
    out = rows['journal/' + profile + '-qualified.stdout']
    err = rows['journal/' + profile + '-qualified.stderr']
    assert all(label in out for label in labels)
    assert b'delta=0 calls=128' in out and b'65536 linked allocations' in out
    assert b'Ran 3 tests' in err and b'Ran 15 tests' in err
    assert b'AddressSanitizer:' not in err and b'runtime error:' not in err
    assert rows['journal/' + profile + '-eval-run-compacted.stdout'].count(b'  Testing ') == 134
    assert rows['journal/' + profile + '-checker-run-compacted.stdout'].count(b'  Testing ') == 88
    assert b'60 passed, 0 failed' in rows['journal/' + profile + '-environment-run-compacted.stdout']
    assert b'I preserve declared-owner identity' in rows['journal/' + profile + '-nominal-run-compacted.stdout']
    assert b'I completed my enum value and repeated-shadow controls' in rows['journal/' + profile + '-enums-run-compacted.stdout']
neighbors = js('neighbors-compacted')
assert len(neighbors) == 20 and all(r['exit'] == 0 for r in neighbors)
for r in neighbors:
    if r['label'].endswith('-run'):
        assert r['detect_leaks'] == (r['label'] not in ['san-eval-run', 'san-checker-run'])
assert js('production')['exit'] == 0
assert not any(a.startswith(('CFLAGS=', 'LDFLAGS=')) for a in js('production')['args'])
assert all(label in rows['journal/production.stdout'] for label in labels)
assert b'Ran 3 tests' in rows['journal/production.stderr'] and b'Ran 15 tests' in rows['journal/production.stderr']
assert js('lexical-controller')['exit'] == 0
q = js('lexical-qualified')
assert q['success'] and q['failures'] == q['errors'] == 0
assert q['test_source_sha256'] == new['tests/test_lexical_scope_boundaries.py']
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
assert js('loaded-save')['exit'] == 0 and js('loaded-save')['saved_bytes'] == 0
assert js('final-image-guards') == json.loads(rows['baseline/image-guards.json'])
for path, digest in js('product-closure').items():
    assert h(rows['provider/' + path]) == digest
assert m['acceptance'] == dict(ordinary_module_rejection_rollback=True,
    copied_enum_owner_retirement=True, borrowed_enum_owner_survival=True,
    first_compiler_source_stage=True, loaded_fragment_save=False, complete_repl=False,
    default_string_activation=False, paired_platform=False, raw_stage=False, whole_release=False)
proc.stdin.close()
assert proc.wait() == 0
print(f'PASS {len(rows)} records / {len(new)} exact Git inputs / {len(images)} native ELF images; '
      'module rollback/recovery and borrowed enum owner controls; '
      'evaluator134/checker88/environment60 plus two1306-shadow source stages; '
      'loaded-save/default/platform/raw/release remain open')
