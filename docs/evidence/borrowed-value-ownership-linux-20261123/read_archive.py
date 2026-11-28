#!/usr/bin/env python3
"""I verify my non-owning view boundary without claiming general reclamation."""
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
    assert manifest['source_pin'].startswith('0dd615a038')
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
    q = json.loads(files['journal/borrow-qualification.json'])
    assert q['source_pin'] == manifest['source_pin']
    for key in ('full_fixture_leak_acceptance','source_borrow_permission_changed','literal_or_field_cleanup_enabled','deadline_changed'):
        assert q[key] is False
    assert q['candidate']['exit'] == 0
    for name,h in q['candidate']['files'].items():
        assert sha(files['source/'+name]) == h
    assert [r['exit'] for r in q['baseline']['commands']] == [0,1,1]
    for label in ('callable','array'):
        report = files['journal/borrow-baseline-'+label+'.stderr']
        assert b'heap-use-after-free' in report
    assert b'env_set_binding_impl' in files['journal/borrow-baseline-callable.stderr']
    assert b'env_reclaim_static_arrays' in files['journal/borrow-baseline-array.stderr']
    rows = q['neighbors']['commands']
    assert len(rows) == 22
    assert all(r['exit'] == (1 if r['label'].endswith('test_typechecker-link') else 0) for r in rows)
    assert [r['exit'] for r in q['corrected_typechecker']] == [0,0,0,0]
    assert b'undefined reference' in files['journal/normal-test_typechecker-link.stderr']
    for mode in ('normal','san'):
        for suffix in ('borrow-probe-run','borrow-array-run'):
            assert files['journal/'+mode+'-'+suffix+'.stderr'] == b''
        assert b'I preserve shared/mutable caller values and exactly-once owned reclamation.' in files['journal/'+mode+'-test-env-borrowed-values.stdout']
        assert files['journal/'+mode+'-test-env-borrowed-values.stderr'] == b''
        assert b'I retain 0 callable names across nine 64-call routes.' in files['journal/'+mode+'-test-eval-callable-lifetimes.stdout']
        assert files['journal/'+mode+'-test-eval-callable-lifetimes.stderr'] == b''
        assert b'All eval tests passed!' in files['journal/'+mode+'-test_eval-run.stdout']
        assert b'All typechecker tests passed!' in files['journal/'+mode+'-typechecker-corrected-run.stdout']
        for row in rows:
            if row['label'].startswith(mode):
                report = files['journal/'+row['label']+'.stderr']
                assert b'ERROR: AddressSanitizer' not in report
                assert b'runtime error:' not in report
    assert q['nvm2c_source_sha256'] == sha(files['source/src/nanoisa/nvm2c.c']) == '994d83e225343cfee19f5674cc30259fcea6bc3db44d65911764fd63489f31c0'
    assert q['nvm2c_image_sha256'] == sha(files['images/bin/nvm2c']) == '007dab36cbffea6c151072155d99d220c2c1aaa3b2ea1847932bba5b00b93d8c'
    for path, identity in q['providers'].items():
        assert identity['bytes'] > 0 and len(identity['sha256']) == 64
    assert files['source/src/eval.c'] == files['journal/baseline-source/src/eval.c']
    assert files['source/src/nanolang.h'] == files['journal/baseline-source/src/nanolang.h']
    assert files['source/src/nanoisa/nvm2c.c'] == files['journal/baseline-source/src/nanoisa/nvm2c.c']
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; borrowed views survive and owned roots retire once')

if __name__ == '__main__':
    main()
