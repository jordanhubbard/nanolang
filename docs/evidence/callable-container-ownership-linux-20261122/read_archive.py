#!/usr/bin/env python3
"""I verify my constructor repair while retaining the full lifetime failures."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['source_pin'].startswith('d732326d70')
    for entry in manifest['base_archives']:
        assert sha((HERE/entry['path']).read_bytes()) == entry['sha256']
    files, objects, descriptors = {}, {}, {}
    for row in manifest['files']:
        assert row['path'] not in files
        descriptor = tuple(row[k] for k in ('artifact', 'bytes', 'compressed_bytes', 'compressed_sha256'))
        if row['sha256'] not in objects:
            path = (HERE/row['artifact']).resolve(strict=True)
            assert path.is_relative_to((ROOT/'docs/evidence').resolve())
            packed = path.read_bytes()
            assert (len(packed), sha(packed)) == (row['compressed_bytes'], row['compressed_sha256'])
            raw = gzip.decompress(packed)
            assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
            objects[row['sha256']] = raw
            descriptors[row['sha256']] = descriptor
        assert descriptors[row['sha256']] == descriptor
        files[row['path']] = objects[row['sha256']]
    counts = manifest['counts']
    assert len(files) == counts['records']
    assert len(objects) == counts['unique_objects']
    assert sum(map(len, files.values())) == counts['logical_bytes']
    own = {Path(r['artifact']).name for r in manifest['files'] if Path(r['artifact']).parent == Path('objects')}
    assert own == {p.name for p in (HERE/'objects').iterdir()}
    paths = [r['path'] for r in manifest['git_sources']]
    tree = subprocess.check_output(['git', 'ls-tree', '-rz', manifest['source_pin'], '--', *paths], cwd=ROOT)
    blobs = {}
    for item in tree.split(b'\0'):
        if item:
            meta, path = item.split(b'\t', 1)
            blobs[path.decode()] = meta.split()[2].decode()
    assert len(blobs) == len(paths) == counts['sources']
    for row in manifest['git_sources']:
        raw = files['source/'+row['path']]
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob'] == blobs[row['path']]
    q = json.loads(files['journal/qualification.json'])
    assert q['source_pin'] == manifest['source_pin']
    assert q['baseline_pin'] == '53abc0a57aebf11f2becead97478e924910caddb'
    assert not q['full_source_fixture_sanitizers_pass'] and not q['strict_lifetimes_pass']
    assert q['nvm2c_unchanged']
    assert sha(files['source/src/env.c']) == q['constructor_source_sha256']
    for name, row in q['baseline']['files'].items():
        raw = files[('images/' if name.startswith('bin/') else 'source/')+name]
        if name == 'src/env.c':
            raw = files['journal/baseline-source/env.c']
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
    assert sha(files['source/src/nanoisa/nvm2c.c']) == '12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67'
    assert sha(files['images/bin/nvm2c']) == 'c94adddd92eefc081ee5cb28df971a36fe2f7cbe8e9730ec68ca6b1c3d24a7ea'
    assert files['source/src/eval.c'] == files['journal/baseline-source/eval.c']
    for prefix in ('baseline', 'test-eval-callable-lifetimes'):
        assert b'I retain 576 callable names across nine 64-call routes.' in files['journal/'+prefix+'.stdout']
    assert q['baseline']['exit'] == 2
    assert [r['exit'] for r in q['candidate']['commands']] == [0, 2]
    assert [r['exit'] for r in q['enhanced']] == [0, 0, 0, 0, 1]
    assert b'actual.as.function_val.function_name != source.as.function_val.function_name' in files['journal/baseline-constructor-run.stdout']
    assert b'heap-use-after-free' in files['journal/escape-asan-run.stderr']
    assert b'eval_call' in files['journal/escape-asan-run.stderr']
    assert b'own_function_identifier' in files['journal/escape-asan-run.stderr']
    for row in q['ordinary']:
        assert row['exit'] == 0
        assert 'baseline' not in row['label']
    assert b'All eval tests passed!' in files['journal/eval-run.stdout']
    assert b'All typechecker tests passed!' in files['journal/typechecker-run.stdout']
    for prefix in ('enhanced-run', 'full-provider-make'):
        text = files['journal/'+prefix+'.stdout']
        assert b'I preserve six independent nested-signature constructor values' in text
        assert b'run every fixture shadow.' in text
    assert [r['exit'] for r in q['sanitizers']['commands']] == [2, 0]
    report = files['journal/full-provider-make.stderr']
    assert b'LeakSanitizer: detected memory leaks' in report
    assert b'3920 byte(s) leaked in 169 allocation(s)' in report
    assert b'heap-use-after-free' not in report
    assert files['journal/full-provider-constructors.stderr'] == b''
    assert b'--constructors-only' in json.dumps(q['sanitizers']).encode()
    # I retain the linked provider code. Individual provider hashes are observed
    # identities, not a claim that every separate object file is archived.
    assert len(q['provider_identities']) > 100
    assert all(raw[:6] == b'\x7fELF\x02\x01' for name, raw in files.items() if name.startswith('products/'))
    print(f"PASS: {len(files)} archive records, {len(paths)} Git inputs; constructor/escape controls pass; 576-name and full-LSan failures retained.")


if __name__ == '__main__':
    main()
