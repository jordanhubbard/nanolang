#!/usr/bin/env python3
"""I verify live tracing snapshots and actual runtime labels."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['source_pin'].startswith('e542ae38f2')
    for row in manifest['base_archives']:
        assert sha((HERE/row['path']).read_bytes()) == row['sha256']
    files, objects, descriptors = {}, {}, {}
    for row in manifest['files']:
        assert row['path'] not in files
        descriptor = tuple(row[k] for k in ('artifact','bytes','compressed_bytes','compressed_sha256'))
        h = row['sha256']
        if h not in objects:
            path = (HERE/row['artifact']).resolve(strict=True)
            assert path.is_relative_to((ROOT/'docs/evidence').resolve())
            packed = path.read_bytes()
            assert (len(packed),sha(packed)) == (row['compressed_bytes'],row['compressed_sha256'])
            raw = gzip.decompress(packed)
            assert (len(raw),sha(raw)) == (row['bytes'],h)
            objects[h], descriptors[h] = raw, descriptor
        assert descriptors[h] == descriptor
        files[row['path']] = objects[h]
    counts = manifest['counts']
    assert len(files) == counts['records']
    assert len(objects) == counts['unique_objects']
    assert sum(map(len,files.values())) == counts['logical_bytes']
    own = {Path(r['artifact']).name for r in manifest['files'] if Path(r['artifact']).parent == Path('objects')}
    assert own == {p.name for p in (HERE/'objects').iterdir()}
    paths = [r['path'] for r in manifest['git_sources']]
    tree = subprocess.check_output(['git','ls-tree','-rz',manifest['source_pin'],'--',*paths],cwd=ROOT)
    blobs = {}
    for item in tree.split(b'\0'):
        if item:
            meta,p = item.split(b'\t',1)
            blobs[p.decode()] = meta.split()[2].decode()
    assert len(blobs) == len(paths) == counts['sources']
    for row in manifest['git_sources']:
        raw = files['source/'+row['path']]
        assert (len(raw),sha(raw)) == (row['bytes'],row['sha256'])
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob'] == blobs[row['path']]
    q = json.loads(files['journal/qualification.json'])
    assert q['source_pin'] == manifest['source_pin']
    for key in ('whole_record_union_leak_acceptance','general_carrier_cleanup_enabled','source_borrow_permission_changed','deadline_changed'):
        assert q[key] is False
    assert q['candidate']['exit'] == 0
    assert q['first_make_refusal']['exit'] == q['strict_enum_refusal']['exit'] == 2
    assert b'missing separator' in files['journal/first-candidate.stderr']
    assert b'enum-conversion' in files['journal/enum-candidate.stderr']
    assert b'old_value' in files['journal/original-baseline-tracing-syntax.stderr']
    assert b'undeclared' in files['journal/original-baseline-tracing-syntax.stderr']
    for path,h in q['candidate']['files'].items():
        assert sha(files['source/'+path]) == h
    assert sha(files['journal/enum-test_eval_assignment_tracing.c']) == q['strict_enum_refusal']['files']['tests/test_eval_assignment_tracing.c']
    assert sha(files['journal/baseline-tracing.c']) == q['strict_enum_refusal']['files']['src/tracing.c']
    rows = q['neighbors']['commands']
    assert len(rows) == 30
    for row in rows:
        assert row['exit'] == (2 if row['label']=='san-test-eval-callable-record-escape' else 1 if row['label']=='late-run' else 0)
    for prefix in ('candidate','normal-test-eval-assignment-tracing'):
        text = files['journal/'+prefix+'.stdout']
        assert text.count(b' VAR_SET ') == 8
        assert text.count(b' FUNCTION_CALL runtime_kind:') == 14
        assert text.count(b'I preserve live assignment values with tracing enabled=') == 3
        assert b'I label all fourteen runtime kinds independently of declaration ordinals.' in text
        assert b'old_value: "before"\n  new_value: "after"' in text
        assert files['journal/'+prefix+'.stderr'] == b''
        for label in ('int','float','bool','string','array','dyn_array','struct','gc_struct','union','function','tuple','coroutine','void','bstring'):
            assert ('name: "input", type: "'+label+'",').encode() in text
    source = files['source/src/eval.c']
    retire = b'            env_set_evaluated_binding(env, binding, value, stmt->as.set.value);\n'
    decl = b'            Value old_value = binding ? binding->value : create_void();\n'
    assert source.count(retire) == source.count(decl) == 1
    assert files['journal/late-eval.c'] == source.replace(retire,b'').replace(decl,decl+retire)
    report = files['journal/late-run.stderr']
    assert b'heap-use-after-free' in report and b'nano_test_trace_var_set' in report
    for mode in ('normal','san'):
        for name in ('test_eval','unwrapped-eval'):
            assert b'All eval tests passed!' in files['journal/'+mode+'-'+name+'-run.stdout']
            assert b'ERROR: AddressSanitizer' not in files['journal/'+mode+'-'+name+'-run.stderr']
        assert b'All typechecker tests passed!' in files['journal/'+mode+'-test_typechecker-run.stdout']
        command = next(r['command'] for r in rows if r['label']==mode+'-unwrapped-eval-compile')
        assert '-DNANO_INTERPRETER' in command
        assert not any('trace_var_set=' in a for a in command)
        assert b'991 allocations, 991 releases.' in files['journal/'+mode+'-test-eval-record-binding-ownership.stdout']
        assert files['journal/'+mode+'-test-eval-callable-expression-ownership.stderr'] == b''
        assert b'I retain 0 callable names across nine 64-call routes.' in files['journal/'+mode+'-test-eval-callable-lifetimes.stdout']
    report = files['journal/san-test-eval-callable-record-escape.stderr']
    assert b'3720 byte(s) leaked in 159 allocation(s)' in report
    assert b'heap-use-after-free' not in report
    base = json.loads((HERE/manifest['base_archives'][0]['path']).read_text())
    prior = {r['path']:r for r in base['git_sources']}
    for path in ('src/env.c','src/tracing.h','src/nanolang.h','src/nanoisa/nvm2c.c','tests/test_eval.c','tests/test_typechecker.c','tests/test_eval_record_binding_ownership.c','tests/test_eval_callable_expression_ownership.c','tests/test_eval_callable_record_escape.c','tests/test_eval_callable_lifetimes.c','tests/test_env_borrowed_values.c','tests/test_env_symbol_index.c'):
        assert sha(files['source/'+path]) == prior[path]['sha256']
    assert q['nvm2c_source_sha256'] == sha(files['source/src/nanoisa/nvm2c.c']) == '994d83e225343cfee19f5674cc30259fcea6bc3db44d65911764fd63489f31c0'
    assert q['nvm2c_image_sha256'] == sha(files['images/bin/nvm2c']) == '007dab36cbffea6c151072155d99d220c2c1aaa3b2ea1847932bba5b00b93d8c'
    for row in base['files']:
        if row['path'].startswith('images/'):
            assert sha(files[row['path']]) == row['sha256']
    assert all(raw.startswith(b'\x7fELF') for path,raw in files.items() if path.startswith('products/'))
    changed = subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines()
    assert set(changed) == {'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CALLABLE_OWNERSHIP.md','src/eval.c','src/tracing.c','tests/test_eval_assignment_tracing.c'}
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; live traces and14 runtime labels; late UAF retained')

if __name__ == '__main__':
    main()
