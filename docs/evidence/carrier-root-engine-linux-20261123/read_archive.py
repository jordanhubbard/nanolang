#!/usr/bin/env python3
"""I verify the private root engine without accepting its pending source adapter."""
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
    assert manifest['source_pin'] == 'c6eb52f54805d0a35f9adbf23c9f5e764cdf9759'
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
    assert manifest['acceptance'] == dict(private_engine=True,source_adapter=False,typed_list_source_fix=False,full_source_lsan=False,whole_release=False)
    q = json.loads(files['journal/qualified.json'])
    for name in ('first-qualified','second-qualified','qualified'):
        report = json.loads(files['journal/'+name+'.json'])
        assert [(r['label'],r['exit']) for r in report['commands']] == [('normal',0),('san',0)]
        for row in report['commands']:
            assert row['command'][-1] == 'test-eval-carrier-heap'
            flags = next(a for a in row['command'] if a.startswith('CFLAGS='))
            assert '-Werror' in flags and '-std=c99' in flags
            assert ('-fsanitize=address,undefined' in flags) == (row['label']=='san')
    for p,h in q['files'].items():
        assert sha(files['source/'+p]) == h
    for mode in ('normal','san'):
        assert files['journal/'+mode+'.stderr'] == b''
        text = files['journal/'+mode+'.stdout']
        assert b'I balance 67664 carrier payload allocations and releases.' in text
        assert b'I preserve replaced storage through an explicit list root, then reclaim it once.' in text
        assert b'I preserve shared and cyclic reachable identities' in text
        assert b'I account for explicit graph transfer, early discard and readoption' in text
        assert b'I refuse duplicate, missing-policy, unknown and reentrant ownership changes' in text
        assert b'I transfer explicit graph ownership between interpreter heaps without copying' in text
        assert b'I trace 65536 linked allocations iteratively' in text
        assert b'I preserve caller ownership through six allocation failures and retire all registry metadata.' in text
        for name in ('first','second'):
            assert files['journal/'+name+'-'+mode+'.stderr'] == b''
            assert b'I balance 67662 carrier payload allocations and releases.' in files['journal/'+name+'-'+mode+'.stdout']
        assert b'through five allocation failures' in files['journal/second-'+mode+'.stdout']
    first = json.loads(files['journal/first-qualified.json'])
    assert sha(files['journal/first-Makefile.gnu']) == first['files']['Makefile.gnu']
    assert sha(files['journal/before-final-test_eval_carrier_heap.c']) == first['files']['tests/test_eval_carrier_heap.c']
    assert files['source/tests/test_eval_carrier_heap_faults.c'] == files['journal/before-final-test_eval_carrier_heap_faults.c'].replace(b'five allocation failures',b'six allocation failures')
    assert b"env['ASAN_OPTIONS']='detect_leaks=1'" in files['journal/observe.py']
    tools = json.loads(files['journal/current-tools.json'])
    assert tools['host'][4] == 'aarch64'
    assert tools['cc']['exit'] == tools['as']['exit'] == 0
    assert '13.3.0' in tools['cc']['stdout'] and '2.42' in tools['as']['stdout']
    guards = json.loads(files['journal/image-guards.json'])
    for path,row in guards.items():
        key = 'images/bin/nanoc_stage2' if path=='bin/nanoc' else 'images/'+path
        assert (len(files[key]),sha(files[key])) == (row['bytes'],row['sha256'])
    assert guards['bin/nanoc']['symlink'] == 'nanoc_stage2'
    base = json.loads((HERE/manifest['base_archives'][0]['path']).read_text())
    prior = {r['path']:r for r in base['git_sources']}
    for path in prior:
        if path not in ('Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md'):
            assert sha(files['source/'+path]) == prior[path]['sha256']
    recipe = b'''.PHONY: test-eval-carrier-heap
test-eval-carrier-heap:
\t@mkdir -p $(OBJ_DIR)
\t$(CC) $(CFLAGS) -o $(OBJ_DIR)/test_eval_carrier_heap tests/test_eval_carrier_heap.c $(SRC_DIR)/eval/carrier_heap.c $(LDFLAGS)
\t$(OBJ_DIR)/test_eval_carrier_heap
\t$(CC) $(CFLAGS) -o $(OBJ_DIR)/test_eval_carrier_heap_faults tests/test_eval_carrier_heap_faults.c $(LDFLAGS)
\t$(OBJ_DIR)/test_eval_carrier_heap_faults

test-units: test-eval-carrier-heap

'''
    make = files['source/Makefile.gnu']
    assert make.count(recipe)==1
    assert sha(make.replace(recipe,b'')) == prior['Makefile.gnu']['sha256']
    assert len(manifest['product_paths']) == 10
    for path in manifest['product_paths']:
        raw = files['products/'+str(Path(path).relative_to('/tmp'))]
        assert raw.startswith(b'\x7fELF') and int.from_bytes(raw[18:20],'little') == 183
    changed = subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines()
    assert set(changed) == {'Makefile.gnu','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md','src/eval/carrier_heap.c','src/eval/carrier_heap.h','tests/test_eval_carrier_heap.c','tests/test_eval_carrier_heap_faults.c'}
    assert b'evaluator/environment/GC paths are still unchanged' in files['source/docs/EVAL_CARRIER_OWNERSHIP.md']
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; ten native products; private engine only, source adapter remains open')

if __name__ == '__main__':
    main()
