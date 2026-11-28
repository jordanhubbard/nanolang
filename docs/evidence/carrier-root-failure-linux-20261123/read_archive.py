#!/usr/bin/env python3
"""I verify actual typed-list root failures without claiming a repair."""
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
    assert manifest['source_pin'] == '1097c4c6a0209da1634d6ccbd3ffaff4290a7876'
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
    assert manifest['acceptance'] == dict(product_repair=False,carrier_lifetime=False,full_source_lsan=False,whole_release=False)
    q = json.loads(files['journal/baseline.json'])
    first = json.loads(files['journal/first-baseline.json'])
    assert q['pin'] == first['pin'] == '1d2e922c48f5f949c1b2ecb157a9b392dfa0adce'
    assert q['provider_source'] == '761f9407fffeb27360e1d233234e335a5d9c28ea'
    expected = [('link',0),('api-control',0),('api-replace',1),('source',1)]
    for observation in (q,first):
        assert [(r['label'],r['exit']) for r in observation['commands']] == expected
    assert q['work'] != first['work']
    for p,h in q['files'].items():
        assert sha(files['source/'+p]) == h
    assert files['journal/api-control.stderr'] == b''
    assert files['journal/api-control.stdout'] == b'I retain the actual typed-list pointer identity before replacement.\nI preserve the retained list record and finish explicit caller teardown.\n'
    assert b'pointer identity before replacement.' in files['journal/api-replace.stdout']
    assert b'finish explicit caller teardown.' not in files['journal/api-replace.stdout']
    for label,read_site in [('api-replace',b'eval_test_record_field'),('source',b'env_define_var_impl')]:
        report = files['journal/'+label+'.stderr']
        assert b'heap-use-after-free' in report and read_site in report
        assert b'freed by thread T0' in report and b'env_set_binding_impl' in report
        assert b'env_free_value' in report and b'allocate_struct_carrier' in report
    source_report = files['journal/source.stderr']
    assert b'"event":"shadow-start","name":"observe_list"' in source_report
    assert b'"event":"shadow-complete"' not in source_report
    assert b'requires unsafe block or unsafe module' in files['journal/first-source.stderr']
    assert b'heap-use-after-free' not in files['journal/first-source.stderr']
    before, after = files['journal/first-source.nano'],files['journal/source.nano']
    assert after == before.replace(b'fn observe_list() -> int {', b'fn observe_list() -> int {\n    unsafe {').replace(b'    return held.number\n}',b'    return held.number\n    }\n}')
    assert after.count(b'shadow ') == 2
    assert b'set item CarrierItem { number: 9, text: "after" }' in after
    assert b'assert (== held.number 7)' in after and b'assert (== held.text "before")' in after
    assert files['journal/probe.c'] == files['journal/first-probe.c']
    assert b'ASAN_OPTIONS' in files['journal/observe.py'] and b'detect_leaks=1' in files['journal/observe.py']
    assert b'not separately' in files['journal/observation-limits.md']
    base = json.loads((HERE/manifest['base_archives'][0]['path']).read_text())
    prior = {r['path']:r for r in base['git_sources']}
    for path in prior:
        if path.startswith(('src/','src_nano/','tests/')) or path == 'Makefile.gnu':
            assert sha(files['source/'+path]) == prior[path]['sha256']
    identities = json.loads(files['journal/input-identities.json'])
    assert identities['provider_source'] == q['provider_source']
    assert len(identities['providers']) == 108
    provider_paths = {arg for arg in q['commands'][0]['command'] if arg.startswith('/tmp/nano-') and arg.endswith(('.o','.a'))}
    assert provider_paths == set(identities['providers'])
    for row in identities['providers'].values():
        assert row['bytes'] > 0 and len(row['sha256']) == 64
    for path,row in identities['images'].items():
        if path.startswith('src/'):
            raw = files['source/'+path]
        else:
            imagepath = 'images/bin/nanoc_stage2' if path == 'bin/nanoc' else 'images/'+path
            raw = files[imagepath]
        assert (len(raw),sha(raw)) == (row['bytes'],row['sha256'])
    assert identities['images']['bin/nanoc']['symlink'] == 'nanoc_stage2'
    for row in base['files']:
        if row['path'].startswith('images/'):
            assert sha(files[row['path']]) == row['sha256']
    assert len(manifest['product_paths']) == 2
    for path in manifest['product_paths']:
        raw = files['products/'+str(Path(path).relative_to('/tmp'))]
        assert raw.startswith(b'\x7fELF') and int.from_bytes(raw[18:20],'little') == 183
    changed = subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',manifest['source_pin']],cwd=ROOT,text=True).splitlines()
    assert set(changed) == {'docs/ROADMAP.md','docs/EVAL_CARRIER_OWNERSHIP.md'}
    assert b'#1211' in files['source/docs/ROADMAP.md']
    assert b'copying list entries or an untracked retention cache' in files['source/docs/ROADMAP.md']
    print('PASS',counts['records'],'records;',counts['sources'],'Git inputs; two native products; API/source use-after-free retained; no product repair')

if __name__ == '__main__':
    main()
