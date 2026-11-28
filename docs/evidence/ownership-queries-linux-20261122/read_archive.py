#!/usr/bin/env python3
"""I verify a partial query checkpoint without accepting its two AOT refusals."""
from pathlib import Path
import gzip
import hashlib
import json
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['source_pin'] == '6e218f2289ace10c9cc35cfe3e4a0021462386db'
    for base in manifest['base_archives']:
        assert sha((HERE/base['path']).read_bytes()) == base['sha256']
    files, objects, descriptors, own_objects = {}, {}, {}, set()
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
                own_objects.add(path.name)
        files[row['path']] = objects[row['sha256']]
    assert manifest['counts'] == dict(records=15689, sources=1371, logical_bytes=207895121, unique_objects=4476)
    assert len(objects) == manifest['counts']['unique_objects']
    assert len(files) == manifest['counts']['records']
    assert sum(map(len, files.values())) == manifest['counts']['logical_bytes']
    assert own_objects == {p.name for p in (HERE/'objects').iterdir()}

    def data(name):
        return json.loads(files[name])

    def lines(name):
        return files[name].decode().splitlines()

    assert isinstance(data('generated/modules/index.json'), dict)
    q = data('qualification.json')
    assert q['source_pin'] == manifest['source_pin']
    assert q['source_compiler_pin'] == '09cab2184f6c889ea8059cf7c5ec29baf456301e'
    assert q['images']['bin/nanoc_stage2']['sha256'] == '346dc0d73af5960dea863fa724a0b1845f5613dff10b8661180ed74bf4eb3d83'
    assert q['full_target_passed'] is False and q['required_aot_refusals'] == 2
    assert q['translator_candidate_compiled'] is False
    for name, row in q['images'].items():
        raw = files['images/'+name.lstrip('/')]
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        if name != '/usr/bin/python3':
            assert raw[:6] == b'\x7fELF\x02\x01'
            assert struct.unpack_from('<H', raw, 18)[0] == 183
    assert sha(files['source/src/nanoisa/nvm2c.c']) == '12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67'
    source_paths = [row['path'] for row in manifest['git_sources']]
    tree = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', manifest['source_pin'], '--', *source_paths], cwd=ROOT)
    blobs = {}
    for item in tree.split(b'\0'):
        if item:
            meta, path = item.split(b'\t', 1)
            mode, kind, identity = meta.split()
            assert kind == b'blob'
            blobs[path.decode()] = identity.decode()
    assert len(blobs) == len(source_paths) == manifest['counts']['sources']
    for row in manifest['git_sources']:
        raw = files['source/'+row['path']]
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob'] == blobs[row['path']]

    observation = data('products/observed/observation.json')
    cases = observation['corpus']
    assert len(cases) == q['corpus']['cases'] == 3080
    assert sum(r['name'].startswith('original-') for r in cases) == q['corpus']['original_declarations']
    names = {row['name'] for row in cases}
    for flags in range(64):
        for op in range(6):
            assert f'pointer-precedence-{flags}-{op}' in names
    for name in ('constructor-full-domain', 'path-full-domain', 'global-full-domain',
                 'effect-full-domain', 'element-full-domain', 'parameter-full-domain',
                 'variant-full-domain', 'present-empty-union-table'):
        assert name in names
    for row in cases:
        raw = files['products/observed/'+row['path']]
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
    reports = lines('products/observed/actual-c.reports')
    assert len(reports) == 3080
    assert sha(files['products/observed/actual-c.reports']) == q['corpus']['complete_report_sha256'] == 'b5973145e3390a20d652a55102bc12b925892c221507c78b261566a7ee9071a0'
    for row, actual in zip(cases, reports):
        if row['expected'] is not None:
            assert actual == row['expected'], row['name']
        if row['expected_status'] is not None:
            assert actual.startswith(str(row['expected_status'])+','), row['name']
        if 'validation-only-union-sentinel' in row['name'] or row['name'] == 'union-range-4294967295-0':
            assert actual == '13,'
    expected_routes = ['actual-c', 'actual-c-sanitized', 'c-seed-native', 'c-producer-vm', 'nanolang-producer-vm']
    assert [r['label'] for r in observation['routes']] == expected_routes
    for row in observation['routes']:
        assert row['reports'] == 3080 and row['sha256'] == q['corpus']['complete_report_sha256']
        assert lines('products/observed/'+row['label']+'.reports') == reports
    commands = observation['commands']
    assert len(commands) == 968
    assert all(r['exit'] == 0 for r in commands[:-2])
    assert all(r['exit'] == 1 for r in commands[-2:])
    for first in (0, 193, 386, 579, 773):
        streamed = []
        for batch, row in enumerate(commands[first:first+193]):
            expected_inputs = [observation['work']+'/'+r['path'] for r in cases[batch*16:batch*16+16]]
            assert row['command'][-len(expected_inputs):] == expected_inputs
            streamed.extend(lines('products/observed/'+row['stdout']))
        assert streamed == reports
    assert commands[772]['command'][0].endswith('/bin/nanoc')
    assert '--emit-nvm' in commands[772]['command']
    for row in commands:
        assert files['products/observed/'+row['stdout']] is not None
        assert files['products/observed/'+row['stderr']] is not None
    assert any('/bin/nanoc' in r['command'][0] and '--emit-nvm' in r['command'] for r in commands)
    for row, name in zip(commands[-2:], ('decoder', 'source-decoder')):
        error = files['products/observed/'+row['stderr']]
        assert b'narr_t versus nmap_value' in error and b'function 8, offset 545' in error
        assert files['products/observed/'+name+'.c'] == b'I preserve prior output on refused C translation\x00\xff\n'
    assert observation['status'] == 'PARTIAL_NATIVE_VM_MATCH_WITH_RETAINED_TWO_AOT_REFUSALS'

    for group, count in [('first', 3070), ('second', 3073), ('final-target', 3080)]:
        original = data('products/'+group+'/manifest.json')
        assert len(original['corpus']) == count
        for row in original['corpus']:
            raw = files['products/'+group+'/'+Path(row['path']).name]
            assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
    old_c = files['observer/attempt-1-source/ownership_contracts.c']
    old_blob = subprocess.check_output(['git', 'rev-parse', '60926b1b992de02a5f40bc51e067e7810c09d91c:src/nanoisa/ownership_contracts.c'], cwd=ROOT, text=True).strip()
    assert hashlib.sha1(b'blob '+str(len(old_c)).encode()+b'\0'+old_c).hexdigest() == old_blob
    new_c = files['source/src/nanoisa/ownership_contracts.c']
    assert new_c.replace(b' ||\n        union_ordinal==NVM_V2_NO_INDEX)', b')', 1) == old_c
    sentinel = data('observer/sentinel-results.json')
    assert [r['label'] for r in sentinel] == ['build-before', 'sentinel-before', 'sentinel-after']
    assert sentinel[0]['exit'] == 0 and sentinel[1]['exit'] != 0 and sentinel[2]['exit'] == 0
    assert b'NVM_V2_ERR_INDEX_RANGE' in files['observer/sentinel-before.stderr']
    assert b'2444 ownership-contract checks' in files['observer/sentinel-after.stdout']
    assert b'Unexpected token in expression: EFFECT' in files['observer/attempt-1.log']
    assert b"'8,' != '13,'" in files['observer/attempt-2.log']
    controls = data('observer/final-controls.json')
    assert [r['exit'] for r in controls] == [2, 0, 0]
    assert b'FAILED (failures=1)' in files['observer/final-full-target.stderr']
    assert b'narr_t versus nmap_value' in files['observer/final-full-target.stderr']
    assert b'Ran 5 tests' in files['observer/identifier-controls.stderr']
    assert b'3446 static fixture fragments; 0 reserved names; 135 lexer-refused' in files['observer/identifier-corpus.stdout']
    assert not files['observer/shadow-policy.log']
    print(json.dumps(dict(status='INTEGRITY_PASS_WITH_REQUIRED_QUERY_AOT_REFUSALS', **manifest['counts']), indent=2))


if __name__ == '__main__':
    main()
