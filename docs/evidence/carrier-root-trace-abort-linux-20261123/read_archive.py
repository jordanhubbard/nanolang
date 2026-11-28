#!/usr/bin/env python3
"""I verify incomplete-trace rollback before accepting a source adapter."""
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
    assert manifest['source_pin'] == '36f7ec1a18061d245e6de80b7f171a40057bb24f'
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
    assert [(r['label'],r['exit']) for r in q['commands']] == [('normal',0),('san',0)]
    for p,h in q['files'].items():
        assert sha(files['source/'+p]) == h
    for row in q['commands']:
        assert row['command'][-1] == 'test-eval-carrier-heap'
        flags = next(a for a in row['command'] if a.startswith('CFLAGS='))
        assert '-Werror' in flags and '-std=c99' in flags
        assert ('-fsanitize=address,undefined' in flags) == (row['label']=='san')
    for mode in ('normal','san'):
        assert files['journal/'+mode+'.stderr'] == b''
        text = files['journal/'+mode+'.stdout']
        for expected in (b'I balance 67667 carrier payload allocations and releases.',
                         b'I preserve replaced storage through an explicit list root, then reclaim it once.',
                         b'I preserve shared and cyclic reachable identities',
                         b'I account for explicit graph transfer, early discard and readoption',
                         b'I refuse duplicate, missing-policy, unknown and reentrant ownership changes',
                         b'I transfer explicit graph ownership between interpreter heaps without copying',
                         b'I abort incomplete tracing before any destruction and recover with the original graph intact.',
                         b'I trace 65536 linked allocations iteratively',
                         b'I preserve caller ownership through six allocation failures and retire all registry metadata.'):
            assert expected in text
    assert b"env['ASAN_OPTIONS']='detect_leaks=1'" in files['journal/observe.py']
    basepath = HERE/manifest['base_archives'][0]['path']
    base = json.loads(basepath.read_text())
    prior = {r['path']:r for r in base['git_sources']}
    changed_paths = {'src/eval/carrier_heap.c','src/eval/carrier_heap.h','tests/test_eval_carrier_heap.c','docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md'}
    for path in prior:
        if path not in changed_paths:
            assert sha(files['source/'+path]) == prior[path]['sha256']
    # I retain every earlier executable assertion while adding rollback/recovery.
    from collections import Counter
    import re
    oldrow = next(r for r in base['files'] if r['path']=='source/tests/test_eval_carrier_heap.c')
    old = gzip.decompress((basepath.parent/oldrow['artifact']).read_bytes())
    assert sha(old) == oldrow['sha256']
    def checks(raw):
        result = []
        for match in re.finditer(rb'\bCHECK\(', raw):
            start = match.end(); end = start; depth = 1
            while depth:
                char = raw[end:end+1]; end += 1
                if char==b'(': depth+=1
                if char==b')': depth-=1
            result.append(raw[start:end-1])
        return Counter(result)
    assert not (checks(old)-checks(files['source/tests/test_eval_carrier_heap.c']))
    guards = json.loads(files['journal/image-guards.json'])
    for path,row in guards.items():
        key = 'images/bin/nanoc_stage2' if path=='bin/nanoc' else 'images/'+path
        assert (len(files[key]),sha(files[key])) == (row['bytes'],row['sha256'])
    assert len(manifest['product_paths']) == 4
    for path in manifest['product_paths']:
        raw = files['products/'+str(Path(path).relative_to('/tmp'))]
        assert raw.startswith(b'\x7fELF') and int.from_bytes(raw[18:20],'little') == 183
    changed = subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines()
    assert set(changed) == changed_paths
    assert b'A tracer returning false aborts before any destruction' in files['source/src/eval/carrier_heap.h']
    assert b'actual source allocation/root/escape producers' in files['source/docs/ROADMAP.md']
    assert b'evaluator/environment/GC paths are still unchanged' in files['source/docs/EVAL_CARRIER_OWNERSHIP.md']
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; four native products; incomplete-trace rollback/recovery; source adapter remains open')

if __name__ == '__main__':
    main()
