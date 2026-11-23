#!/usr/bin/env python3
"""I verify scalar snapshots and retain complete carrier leak failures."""
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
    assert manifest['source_pin'] == '761f9407fffeb27360e1d233234e335a5d9c28ea'
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
    assert manifest['acceptance'] == dict(scalar_projection_api=True, source_semantics=True, whole_source_lsan=False, whole_record_union_lsan=False, whole_release=False)
    q = json.loads(files['journal/final-qualified.json'])
    assert len(q['commands']) == 27
    for row in q['commands']:
        expected = 2 if row['label'] in ('san-test-eval-scalar-projection-sources','san-test-eval-callable-record-escape') else 0
        assert row['exit'] == expected
    for path,h in q['files'].items():
        assert sha(files['source/'+path]) == h
    obs = json.loads(files['journal/final-observations.json'])
    assert [(r['label'],r['exit']) for r in obs] == [('normal-shadow-observation',0),('san-shadow-observation',0),('old-api-link',0),('old-api-run',1)]
    for name,expected in [('baseline',[0,0]),('routes-baseline',[0,1,1]),('matched-tuple',[0,0]),('prefix-try-baseline',[0,1]),('api-corrected',[0,0])]:
        assert [r['exit'] for r in json.loads(files['journal/'+name+'.json'])['commands']] == expected
    assert b'heap-use-after-free' in files['journal/tuple.stderr']
    assert b'fixed resource records with numeric or boolean fields' in files['journal/store.stderr']
    assert b'result.as.string_val != source.as.string_val' in files['journal/old-api-run.stdout'] + files['journal/old-api-run.stderr']
    assert b'1937 byte(s) leaked in 82 allocation(s)' in files['journal/san-test-eval-scalar-projection-sources.stderr']
    assert b'3720 byte(s) leaked in 159 allocation(s)' in files['journal/san-test-eval-callable-record-escape.stderr']
    assert b'heap-use-after-free' not in files['journal/san-test-eval-scalar-projection-sources.stderr']
    assert b'let-name=text rhs-type=5' in files['journal/prefix-try-baseline-run.stderr']
    assert b'name=text kind=12 decl=4' in files['journal/prefix-try-baseline-run.stderr']
    names = {'tuple_case','try_case','try_error_case','unwrap_case','unwrap_error_case','fallback_case','main'}
    for mode in ('normal','san'):
        stdout = files['journal/'+mode+'-test-eval-scalar-projection-ownership.stdout']
        assert stdout.count(b'I preserve ') == 16
        assert b'I balance scalar snapshot metadata: 29 allocations, 29 releases.' in stdout
        assert files['journal/'+mode+'-test-eval-scalar-projection-ownership.stderr'] == b''
        events = [json.loads(l) for l in files['journal/'+mode+'-shadow-observation.stderr'].splitlines() if l.startswith(b'{')]
        assert len(events) == 14 and all(e['failures']==0 for e in events)
        for event in ('shadow-start','shadow-complete'):
            assert {e['name'] for e in events if e['event']==event} == names
            assert sum(e['event']==event for e in events)==7
        text = files['journal/'+mode+'-test_eval-run.stdout']
        assert b'All eval tests passed!' in text and text.count(b'  Testing ') == 134
        text = files['journal/'+mode+'-test_typechecker-run.stdout']
        assert b'All typechecker tests passed!' in text and text.count(b'  Testing ') == 88
        assert b'991 allocations, 991 releases.' in files['journal/'+mode+'-test-eval-record-binding-ownership.stdout']
        assert files['journal/'+mode+'-test-eval-callable-expression-ownership.stderr'] == b''
        assert b'I retain 0 callable names across nine 64-call routes.' in files['journal/'+mode+'-test-eval-callable-lifetimes.stdout']
    assert b'ASAN_OPTIONS' in files['journal/final-qualify.py'] and b'detect_leaks=0' in files['journal/final-qualify.py']
    base = json.loads((HERE/manifest['base_archives'][0]['path']).read_text())
    prior = {r['path']:r for r in base['git_sources']}
    for path in ('src/env.c','src/nanolang.h','src/parser.c','src/typechecker.c','src/nanoisa/nvm2c.c','src/eval/eval_io.c','src/eval/record_fields.inc','tests/test_eval.c','tests/test_typechecker.c','tests/test_eval_callable_record_escape.c'):
        assert sha(files['source/'+path]) == prior[path]['sha256']
    for row in base['files']:
        if row['path'].startswith('images/'):
            assert sha(files[row['path']]) == row['sha256']
    assert len(manifest['product_paths']) >= 22
    for path in manifest['product_paths']:
        raw = files['products/'+str(Path(path).relative_to('/tmp'))]
        assert raw.startswith(b'\x7fELF') and int.from_bytes(raw[18:20],'little') == 183
    changed = subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines()
    assert set(changed) == {'src/eval.c','Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CALLABLE_OWNERSHIP.md','tests/test_eval_scalar_projection_ownership.c','tests/test_eval_scalar_projection_sources.c','tests/fixtures/scalar_projection_ownership.nano','tests/fixtures/scalar_projection_invalid_heap_borrow.nano'}
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs;',len(manifest['product_paths']),'native products; scalar snapshots pass, whole carrier ownership remains open')

if __name__ == '__main__':
    main()
