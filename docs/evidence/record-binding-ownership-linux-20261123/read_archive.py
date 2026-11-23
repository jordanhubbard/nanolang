#!/usr/bin/env python3
"""I verify copied binding retirement and retain the remaining carrier failures."""
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
    assert manifest['source_pin'].startswith('fead5bf2d9')
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
    for key in ('whole_record_union_leak_acceptance','tracing_enabled_acceptance','general_carrier_cleanup_enabled','source_borrow_permission_changed','deadline_changed'):
        assert q[key] is False
    assert [r['exit'] for r in q['qualified']['commands']] == [0,0,2,0,0,0]
    assert [r['exit'] for r in q['neighbors']['commands']] == [0]*15+[1,1]
    assert [r['exit'] for r in q['baseline']['commands']] == [1,1]
    for name,h in q['qualified']['files'].items():
        assert sha(files['source/'+name]) == h
    for row in q['baseline']['commands']:
        assert sha(files['products/baseline-'+row['label']]) == row['image_sha256']
    for name,expected in (('expression',b'572 byte(s) leaked in 26 allocation(s)'),('record',b'3720 byte(s) leaked in 159 allocation(s)')):
        assert expected in files['journal/'+name+'-baseline.stderr']
    for prefix in ('','normal-'):
        report = files['journal/'+prefix+'test-eval-record-binding-ownership.stdout']
        assert b'991 allocations, 991 releases.' in report
        assert b'legacy and raw-array cleanup.' in report
        assert b'nested record lifetime and shared array identity' in report
        assert files['journal/'+prefix+'test-eval-record-binding-ownership.stderr'] == b''
        assert files['journal/'+prefix+'test-eval-callable-expression-ownership.stderr'] == b''
        assert b'I retain 0 callable names across nine 64-call routes.' in files['journal/'+prefix+'test-eval-callable-lifetimes.stdout']
    assert b'991 allocations, 394 releases.' in files['journal/old-providers-run.stdout']
    assert b'FAILED: s_metadata_allocated == s_metadata_released' in files['journal/old-providers-run.stdout']
    report = files['journal/test-eval-callable-record-escape.stderr']
    assert b'3720 byte(s) leaked in 159 allocation(s)' in report
    assert b'heap-use-after-free' not in report
    for mode in ('normal','san'):
        assert b'All eval tests passed!' in files['journal/'+mode+'-test_eval-run.stdout']
        assert b'All typechecker tests passed!' in files['journal/'+mode+'-test_typechecker-run.stdout']
        for suffix in ('test_eval-run','test_typechecker-run'):
            report = files['journal/'+mode+'-'+suffix+'.stderr']
            assert b'ERROR: AddressSanitizer' not in report and b'runtime error:' not in report
    trace = files['journal/baseline-tracing-syntax.stderr']
    assert b'old_value' in trace and b'undeclared' in trace
    assert files['journal/baseline-eval.c'] != files['source/src/eval.c']
    base = json.loads((HERE/manifest['base_archives'][0]['path']).read_text())
    prior = {r['path']:r for r in base['git_sources']}
    for path in ('src/nanolang.h','src/nanoisa/nvm2c.c','src/eval/record_carrier_cleanup.inc','tests/test_eval.c','tests/test_typechecker.c','tests/test_eval_callable_expression_ownership.c','tests/test_eval_callable_record_escape.c','tests/test_eval_callable_lifetimes.c','tests/test_env_borrowed_values.c'):
        assert sha(files['source/'+path]) == prior[path]['sha256']
    assert sha(files['journal/baseline-eval.c']) == prior['src/eval.c']['sha256']
    path = 'tests/test_env_symbol_index.c'
    raw = files['source/'+path]
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert blob == subprocess.check_output(['git','rev-parse','9b604958e0:'+path],cwd=ROOT,text=True).strip()
    assert b'KeyError' in files['journal/first-reader.stderr']
    assert q['nvm2c_source_sha256'] == sha(files['source/src/nanoisa/nvm2c.c']) == '994d83e225343cfee19f5674cc30259fcea6bc3db44d65911764fd63489f31c0'
    assert q['nvm2c_image_sha256'] == sha(files['images/bin/nvm2c']) == '007dab36cbffea6c151072155d99d220c2c1aaa3b2ea1847932bba5b00b93d8c'
    assert all(raw.startswith(b'\x7fELF') for path,raw in files.items() if path.startswith('products/'))
    changed = subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines()
    assert set(changed) == {'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CALLABLE_OWNERSHIP.md','src/env.c','src/eval.c','tests/test_eval_record_binding_ownership.c'}
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; full expression leak gate passes; full record failure retained')

if __name__ == '__main__':
    main()
