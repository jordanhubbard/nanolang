#!/usr/bin/env python3
"""I verify a partial CFG checkpoint without treating retained refusals as passes."""
from pathlib import Path
import gzip
import hashlib
import json
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['source_pin'] == '45223340186ae620fad7ed6272b13fb497320e8a'
    for base in manifest['base_archives']:
        assert sha((HERE/base['path']).read_bytes()) == base['sha256']
    files, descriptors, decoded = {}, {}, {}
    own_objects = set()
    for row in manifest['files']:
        assert row['path'] not in files
        descriptor = tuple(row[k] for k in ('artifact', 'bytes', 'compressed_bytes', 'compressed_sha256'))
        if row['sha256'] in descriptors:
            assert descriptor == descriptors[row['sha256']]
        else:
            path = (HERE/row['artifact']).resolve(strict=True)
            assert path.is_relative_to((ROOT/'docs/evidence').resolve())
            packed = path.read_bytes()
            assert (len(packed), sha(packed)) == (row['compressed_bytes'], row['compressed_sha256'])
            raw = gzip.decompress(packed)
            assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
            descriptors[row['sha256']] = descriptor
            decoded[row['sha256']] = raw
            if path.parent == HERE/'objects':
                own_objects.add(path.name)
        files[row['path']] = decoded[row['sha256']]
    assert own_objects == {p.name for p in (HERE/'objects').iterdir()}
    assert len(files) == manifest['counts']['records'] == 4576
    assert sum(len(b) for b in files.values()) == manifest['counts']['logical_bytes']

    def data(name):
        return json.loads(files[name])

    def lines(name):
        return files[name].decode().splitlines()

    q = data('qualification.json')
    assert q['source_pin'] == manifest['source_pin']
    assert q['source_compiler_pin'] == '09cab2184f6c889ea8059cf7c5ec29baf456301e'
    assert q['images']['bin/nanoc_stage2']['sha256'] == '346dc0d73af5960dea863fa724a0b1845f5613dff10b8661180ed74bf4eb3d83'
    for name, image in q['images'].items():
        raw = files['images/'+name.lstrip('/')]
        assert (len(raw), sha(raw)) == (image['bytes'], image['sha256'])
        if name not in ('/usr/bin/python3',):
            assert raw[:4] == b'\x7fELF' and raw[4:6] == b'\x02\x01'
            assert struct.unpack_from('<H', raw, 18)[0] == 183
    assert q['controls'] == dict(shape_checks=3890, identifier_methods=5, identifier_fragments=3442,
                                 lexer_refused_fragments=135, shadow_policy_exit=0)
    assert b'Shape constraints: 3890 passed, 0 failed' in files['observer/controls.stdout']
    assert b'I lexed 3442 static fixture fragments; 0 reserved names; 135 lexer-refused' in files['observer/controls.stdout']
    assert b'Ran 5 tests' in files['observer/controls.stderr']
    assert files['observer/controls.stderr'].rstrip().endswith(b'OK')
    assert not files['observer/shadow-policy.stderr']

    source_paths = [row['path'] for row in manifest['git_sources']]
    tree = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', manifest['source_pin'], '--', *source_paths], cwd=ROOT)
    blobs = {}
    for item in tree.split(b'\0'):
        if item:
            meta, path = item.split(b'\t', 1)
            mode, kind, identity = meta.split()
            assert kind == b'blob'
            blobs[path.decode()] = identity.decode()
    assert len(blobs) == len(source_paths) == manifest['counts']['sources'] == 1364
    for row in manifest['git_sources']:
        raw = files['source/'+row['path']]
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob'] == blobs[row['path']]
    assert manifest['generated_inputs'] == ['generated/modules/index.json']
    assert isinstance(data('generated/modules/index.json'), dict)

    # I retain the entire required frame corpus, including its failed route construction.
    frame = data('products/frame/manifest.json')
    assert (frame['flow_operations'], frame['normalized_flow_states'], frame['queue_traces'], frame['frame_cases']) == (168386, 289, 7, 665)
    assert len(frame['corpus']) == 1001 and len(frame['routes']) == 1
    assert frame['routes'][0]['label'] == 'actual-c'
    expected_frame = lines('products/frame/actual-c.reports')
    assert len(expected_frame) == 1001
    for index, row in enumerate(frame['corpus']):
        raw = files[f'products/frame/{index:04d}.bin']
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        if row['name'].startswith(('global-algebra-', 'bounded-fifo-')):
            assert expected_frame[index] == row['expected']
        else:
            assert expected_frame[index].startswith(row['expected'])
    assert lines('products/first-parse/actual-c.reports') == expected_frame
    assert b'Expected' in files['observer/first.stderr'] and b'Parsing error' in files['observer/first.stderr']
    assert b'I cannot widen an exactly constrained bool destination' in files['observer/second.stderr']
    assert b'I cannot widen an exactly constrained bool destination' in files['observer/producer-localize-1.stderr']
    assert b'I cannot widen an exactly constrained bool destination' in files['observer/producer-localize-3.stderr']
    localization = data('observer/producer-localize.json')
    assert [row['exit'] for row in localization] == [0, 1, 0, 1]
    assert b'ARRAY_READ 4' in files['observer/shape-map-complete.stdout']
    assert b'LOCAL 6 stack_meet 28' in files['observer/shape-map-complete.stdout']
    assert b'Expected' not in files['observer/second.stderr']
    prior = json.loads((HERE/'verification/prior-output.json').read_text())
    assert prior['source_pin'] == manifest['source_pin']
    assert prior['manifest_sha256'] == sha((HERE/'manifest.json').read_bytes())
    assert prior['tool_sha256'] == q['images']['bin/nvm2c']['sha256']
    assert [r['label'] for r in prior['cases']] == ['c-producer', 'nanolang-producer']
    for row in prior['cases']:
        assert sha(files[row['module_archive']]) == row['module_sha256']
        before, after = bytes.fromhex(row['prior_hex']), bytes.fromhex(row['after_hex'])
        assert before == after == b'I retain this original output.\0\xff\n'
        assert sha(before) == row['prior_sha256'] == row['after_sha256']
        assert row['exit'] == 1 and not bytes.fromhex(row['stdout_hex'])
        assert 'I cannot widen an exactly constrained bool destination' in row['stderr']

    observed_routes = data('observer/constructed-result.json')['routes']+data('observer/pure-result.json')['routes']
    assert [r['label'] for r in observed_routes] == ['constructed-c-native', 'constructed-c-vm', 'constructed-nanolang-vm']
    for route in observed_routes:
        label = route['label']
        reports = []
        for start in range(0, 1001, 8):
            reports += lines(f'observer/{label}-{start:04d}.stdout')
        assert reports == expected_frame and route['reports'] == 1001
        assert sha(('\n'.join(reports)+'\n').encode()) == route['sha256']
    for log in ('constructed-progress.jsonl', 'pure-progress.jsonl'):
        progress = [json.loads(row) for row in lines('observer/'+log)]
        assert progress and all(row['exit'] == 0 and row['match'] for row in progress)

    flow = data('products/flow/manifest.json')
    assert flow['flow_operations'] == 168386 and flow['normalized_flow_states'] == 289
    assert len(flow['corpus']) == 336 and len(flow['routes']) == 6
    assert {r['label'] for r in flow['routes']} == {'actual-c', 'c-seed-native', 'c-producer-nanoisa-vm',
            'c-producer-sanitized-native', 'nanolang-producer-nanoisa-vm', 'nanolang-producer-sanitized-native'}
    expected_flow = [row['expected'] for row in flow['corpus']]
    for index, row in enumerate(flow['corpus']):
        raw = files[f'products/flow/{index:04d}.bin']
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
    for route in flow['routes']:
        raw = files['products/flow/'+route['label']+'.reports']
        assert raw.decode().splitlines() == expected_flow
        assert sha(raw) == route['sha256'] and route['reports'] == 336
    assert b'Ran 1 test in 218.903s' in files['observer/flow.stderr']
    assert files['observer/flow.stderr'].rstrip().endswith(b'OK')
    result = dict(status='INTEGRITY_PASS_WITH_RETAINED_FRAME_AOT_REFUSALS', source_pin=manifest['source_pin'],
                  records=len(files), sources=len(source_paths), flow_operations=168386,
                  normalized_flow_states=289, flow_carriers=336, flow_source_routes=5,
                  frame_carriers=1001, frame_cases=665, frame_source_routes_observed=3,
                  frame_five_route_acceptance=False, refused_prior_outputs_preserved=True, whole_function_cfg=False,
                  current_darwin=False, pure_emission=False, release_acceptance=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
