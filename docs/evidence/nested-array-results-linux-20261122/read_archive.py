#!/usr/bin/env python3
"""I verify my reduced pre-shape barrier without accepting its four refusals."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['source_pin'] == '57696a367262ab67d56fa212cd9e4bc724ae0c8f'
    assert manifest['counts'] == dict(records=1992, sources=1372, logical_bytes=506461980, unique_objects=1534)
    for row in manifest['base_archives']:
        assert sha((HERE/row['path']).read_bytes()) == row['sha256']
    files, objects, descriptors, own = {}, {}, {}, set()
    for row in manifest['files']:
        assert row['path'] not in files
        descriptor = tuple(row[k] for k in ('artifact', 'bytes', 'compressed_bytes', 'compressed_sha256'))
        if row['sha256'] in descriptors:
            assert descriptors[row['sha256']] == descriptor
        else:
            path = (HERE/row['artifact']).resolve(strict=True)
            assert path.is_relative_to((ROOT/'docs/evidence').resolve())
            packed = path.read_bytes()
            assert (len(packed), sha(packed)) == (row['compressed_bytes'], row['compressed_sha256'])
            raw = gzip.decompress(packed)
            assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
            descriptors[row['sha256']] = descriptor
            objects[row['sha256']] = raw
            if path.parent == HERE/'objects':
                own.add(path.name)
        files[row['path']] = objects[row['sha256']]
    assert len(files) == 1992 and len(objects) == 1534
    assert sum(map(len, files.values())) == 506461980
    assert own == {p.name for p in (HERE/'objects').iterdir()}

    def data(name):
        return json.loads(files[name])

    q = data('qualification.json')
    assert q['source_pin'] == manifest['source_pin']
    assert q['source_compiler_pin'] == '09cab2184f6c889ea8059cf7c5ec29baf456301e'
    assert q['files']['bin/nanoc_stage2']['sha256'] == '346dc0d73af5960dea863fa724a0b1845f5613dff10b8661180ed74bf4eb3d83'
    assert q['caller_patch_applied'] is False and q['caller_candidate_compiled'] is False
    assert (q['build_seconds'], q['run_seconds']) == (600, 600)
    assert sha(files['source/src/nanoisa/nvm2c.c']) == '12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67'
    for name, row in q['files'].items():
        if name.startswith(('src/', 'tests/')):
            raw = files['source/'+name]
        else:
            raw = files['images/'+name.lstrip('/')]
            assert raw[:6] == b'\x7fELF\x02\x01' and struct.unpack_from('<H', raw, 18)[0] == 183
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
    assert isinstance(data('generated/modules/index.json'), dict)
    paths = [r['path'] for r in manifest['git_sources']]
    tree = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', manifest['source_pin'], '--', *paths], cwd=ROOT)
    blobs = {}
    for item in tree.split(b'\0'):
        if item:
            meta, path = item.split(b'\t', 1)
            mode, kind, identity = meta.split()
            assert kind == b'blob'
            blobs[path.decode()] = identity.decode()
    assert len(blobs) == len(paths) == 1372
    for row in manifest['git_sources']:
        raw = files['source/'+row['path']]
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob'] == blobs[row['path']]

    # I distinguish valid-source refusals from expected wrong-element refusals.
    for group in ('first-regression', 'regression'):
        prefix = 'products/'+group+'/'
        record = data(prefix+'manifest.json')
        assert len(record['cases']) == 8 and len(record['commands']) == 92
        labels = {}
        for i, row in enumerate(record['commands']):
            assert row['label'] not in labels
            labels[row['label']] = row
            assert sha(files[prefix+f'{i:04d}.stdout']) == row['stdout_sha256']
            assert sha(files[prefix+f'{i:04d}.stderr']) == row['stderr_sha256']
            if row['label'].endswith(('-legacy-run', '-native-run', '-vm')):
                assert row['exit'] == 0 and files[prefix+f'{i:04d}.stdout'] == b'9\n'
            if row['exit'] and row['label'].endswith('-aot'):
                assert b'conflicting kinds' in files[prefix+f'{i:04d}.stderr']
                assert b'result field 0 of function 1' in files[prefix+f'{i:04d}.stderr']
            if row['label'].endswith('-native-build'):
                assert '-fsanitize=address,undefined' in row['command']
                assert '-fno-sanitize-recover=all' in row['command']
        assert sum(row['exit'] == 0 for row in record['commands']) == 84
        assert sum(row['exit'] == 1 for row in record['commands']) == 8
        for suffix, count in [('-legacy-run', 8), ('-vm', 16), ('-native-run', 12)]:
            assert sum(row['label'].endswith(suffix) for row in record['commands']) == count
        for row in record['commands']:
            if row['exit']:
                assert row['label'].startswith('wrong-element-kind-') or (
                    row['label'].startswith(('nested1-slice0-fallback1-', 'nested1-slice1-fallback1-'))
                    and row['label'].endswith('-aot'))
        for case in record['cases']:
            name = case['name']
            raw = files[prefix+name+'.nano']
            assert sha(raw) == case['source_sha256']
            text = raw.decode()
            assert set(re.findall(r'^fn (\w+)', text, re.M)) <= set(re.findall(r'^shadow (\w+)', text, re.M))
            assert {r['label'] for r in case['producers']} == {'c-producer', 'nanolang-producer'}
            failed = name.startswith('nested1-') and name.endswith('fallback1')
            for product in case['producers']:
                stem = name+'-'+product['label']
                assert files[prefix+stem+'.nvm'][:4] == b'NVM\x02'
                assert sha(files[prefix+stem+'.nvm']) == product['module_sha256']
                assert product['aot_exit'] == int(failed)
                assert product['prior_preserved'] == failed
                if failed:
                    assert files[prefix+stem+'.c'] == b'I retain my prior output.\0\xff\n'
        for nested in (0, 1):
            for producer in ('c-producer', 'nanolang-producer'):
                name = f'wrong-element-kind-nested{nested}-{producer}'
                assert labels[name]['exit'] == 1
                assert files[prefix+name+'.nvm'] == b'I retain my prior output.\0\xff\n'
    final = data('products/regression/manifest.json')
    extra = b' assert (== (array_length (select_rows Data { rows: [] }).fields) 0)\n'
    for case in final['cases']:
        name = case['name']; raw = files['products/regression/'+name+'.nano']
        first = files['products/first-regression/'+name+'.nano']
        assert raw.count(extra) == int(name.endswith('fallback1'))
        assert raw.replace(extra, b'') == first == files['products/original/'+name+'.nano']
    original = data('observer/result.json')
    assert len(original['cases']) == 8 and len(original['commands']) == 24
    assert sum(c['aot'] == 1 for c in original['cases']) == 2
    assert all(c['producer'] == 0 for c in original['cases'])
    for name, function, offset, discovery in [
        ('nested1-slice0-fallback1', 1, 59, 0),
        ('nested1-slice1-fallback1', 1, 208, 0),
        ('full-query', 8, 545, 1),
    ]:
        text = files['observer/'+name+'.gdb.stdout'].decode()
        assert f'$1 = {function}\n$2 = {offset}\n$3 = 0\n$4 = false\n' in text
        assert f'discover_globals = {discovery}, final = 0' in text
        assert "$6 = 3 '\\003'" in text and "$7 = 8 '\\b'" in text
        assert all(fn in text for fn in ('merge_record_results', 'merge_parameter', 'merge_fact'))
    assert all(r['prior_preserved'] for r in data('observer/debugger.json'))
    assert data('observer/full-query-debugger.json')['prior_preserved']
    assert (HERE/'reproduce.nano').read_bytes() == files['products/original/nested1-slice0-fallback1.nano']
    for name in ('regression.stderr', 'first-regression.stderr'):
        assert b'FAILED (failures=4)' in files['observer/'+name]
    assert b'Ran 5 tests' in files['observer/final-identifier-controls.stderr']
    assert b'3451 static fixture fragments; 0 reserved names; 135 lexer-refused' in files['observer/final-identifier-corpus.stdout']
    assert not files['observer/final-identifier-corpus.stderr']
    print(json.dumps(dict(status='LOCALIZATION_INTEGRITY_PASS_WITH_FOUR_REQUIRED_AOT_REFUSALS', **manifest['counts']), indent=2))


if __name__ == '__main__':
    main()
