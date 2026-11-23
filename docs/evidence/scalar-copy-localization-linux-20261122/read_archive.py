#!/usr/bin/env python3
"""I verify localization evidence while requiring the production regression to stay red."""
from pathlib import Path
import difflib
import gzip
import hashlib
import json
import struct
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['source_pin'] == '0b81a823608e923d7c989be064a40d8e54d24a63'
    for row in manifest['base_archives']:
        assert sha((HERE/row['path']).read_bytes()) == row['sha256']
    base_path = HERE/manifest['base_archives'][0]['path']
    base = json.loads(base_path.read_text())
    base_files = {r['path']: r for r in base['files']}
    descriptors, decoded, files, own = {}, {}, {}, set()
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
                own.add(path.name)
        files[row['path']] = decoded[row['sha256']]
    assert own == {p.name for p in (HERE/'objects').iterdir()}
    assert len(files) == manifest['counts']['records'] == 2790
    assert sum(len(raw) for raw in files.values()) == manifest['counts']['logical_bytes'] == 101770256

    def data(name):
        return json.loads(files[name])

    q = data('qualification.json')
    assert not q['caller_patch_applied'] and not q['caller_candidate_compiled']
    assert q['source_pin'] == manifest['source_pin']
    assert q['images']['bin/nvm2c']['sha256'] == base_files['images/bin/nvm2c']['sha256']
    assert sha(files['original/frame-decoder.nvm']) == base_files['products/frame/decoder.nvm']['sha256']
    source_paths = [r['path'] for r in manifest['git_sources']]
    tree = subprocess.check_output(['git', 'ls-tree', '-rz', '--full-tree', manifest['source_pin'], '--', *source_paths], cwd=ROOT)
    blobs = {}
    for item in tree.split(b'\0'):
        if item:
            meta, name = item.split(b'\t', 1)
            mode, kind, identity = meta.split()
            assert kind == b'blob'
            blobs[name.decode()] = identity.decode()
    assert len(blobs) == len(source_paths) == manifest['counts']['sources'] == 1366
    for row in manifest['git_sources']:
        raw = files['source/'+row['path']]
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == row['git_blob'] == blobs[row['path']]
    for name, digest in q['protected_sources'].items():
        assert sha(files['source/'+name]) == digest == base_files['source/'+name]['sha256']
    for name, image in q['images'].items():
        raw = files['images/'+name.lstrip('/')]
        assert (len(raw), sha(raw)) == (image['bytes'], image['sha256'])
    assert b'Inferred scalar copy graphs: 312 passed, 0 failed' in files['observer/shape-api.stdout']
    assert b'Inferred scalar copy graphs: 312 passed, 0 failed' in files['observer/shape-sanitizer-1.stdout']
    assert not files['observer/shape-api.stderr']
    assert not files['observer/shape-sanitizer-1.stderr']
    san = data('observer/guards-and-sanitizer.json')
    assert san['production_source_unchanged'] and all(r['exit'] == 0 for r in san['commands'])
    assert b'__asan_init' in files['observer/shape-api-sanitized']
    assert b'__ubsan_handle_' in files['observer/shape-api-sanitized']
    assert b'Ran 5 tests' in files['observer/final-identifiers.stderr']
    assert files['observer/final-identifiers.stderr'].rstrip().endswith(b'OK')
    assert b'I lexed 3446 static fixture fragments; 0 reserved names; 135 lexer-refused' in files['observer/final-identifiers.stdout']

    expected_red = {f'called0-slice{slice_}-write1-guard{guard}-{operation}'
                    for slice_ in (0, 1) for guard in (0, 1) for operation in ('and', 'or')}
    basic = data('observer/result.json')
    copies = data('observer/duplicate-result.json')
    assert len(basic['cases']) == 16 and len(copies['cases']) == 48
    for group, collection in [('basic', basic), ('copy', copies)]:
        failed = set()
        for row in collection['cases']:
            name = row['name']
            assert sha(files[f'products/{group}/{name}.nano']) == row['source_sha256']
            assert [r['label'] for r in row['commands']] == ['producer', 'aot']
            assert row['commands'][0]['exit'] == 0
            if row['commands'][1]['exit']:
                failed.add(name)
                assert row['commands'][1]['exit'] == 1
                assert b'I cannot widen an exactly constrained bool destination' in files[f'products/{group}/{name}-aot.stderr']
        assert failed == (expected_red if group == 'copy' else set())

    regression = data('products/regression/manifest.json')
    assert len(regression['cases']) == 64 and len(regression['commands']) == 134
    assert {r['name'] for r in regression['cases'] if r['aot_exit']} == expected_red
    prior = b'I retain my prior C output.\0\xff\n'
    for row in regression['cases']:
        name = row['name']
        assert sha(files[f'products/regression/{name}.nano']) == row['source_sha256']
        assert sha(files[f'products/regression/{name}.nvm']) == row['module_sha256']
        output = files[f'products/regression/{name}.c']
        assert sha(output) == row['output_sha256']
        assert row['prior_preserved'] == (name in expected_red)
        assert (output == prior) == row['prior_preserved']
    for index, row in enumerate(regression['commands']):
        for suffix in ('stdout', 'stderr'):
            assert sha(files[f'products/regression/{index:04d}.{suffix}']) == row[suffix+'_sha256']
    commands = {r['label']: (i, r) for i, r in enumerate(regression['commands'])}
    for label, expected in [('pick-0-assembler', 0), ('pick-0-verify', 0), ('pick-0-aot', 1),
                            ('pick-1-assembler', 1), ('pick-1-verify', 1), ('pick-1-aot', 1)]:
        index, row = commands[label]
        assert row['exit'] == expected
        if label in ('pick-1-assembler', 'pick-1-verify'):
            assert b'stack underflow' in files[f'products/regression/{index:04d}.stderr']
    for depth in (0, 1):
        assert files[f'products/regression/pick-{depth}.c'] == prior
    valid = files['products/regression/pick-0.nvm']
    invalid = files['products/regression/pick-1.nvm']
    mutation = regression['negative_mutation']
    at = mutation['byte_offset']
    assert valid[at-1:at+2] == b'\xf4\0\0' and invalid[at-1:at+2] == b'\xf4\1\0'
    assert sha(invalid) == mutation['sha256']
    assert invalid[:4] == valid[:4] == b'NVM\x02'
    for raw in (valid, invalid):
        assert zlib.crc32(raw[40:]) & 0xffffffff == struct.unpack_from('<I', raw, 36)[0]
    assert {i for i, (a, b) in enumerate(zip(valid, invalid)) if a != b} <= {36, 37, 38, 39, at}
    assert b'FAILED (failures=9)' in files['observer/corrected-regression.stderr']
    assert b'FAILED (failures=10)' in files['observer/first-regression.stderr']

    order = files['observer/order.stdout']
    exact = files['observer/exact-order.stdout']
    assert b'ARRAY_READ_REGISTRATION element=1600 result=1675' in order
    assert b'LOCAL_READ_UNIFICATION a=1675 b=1662' in order
    assert b'NVM_SHAPE_UNKNOWN' in order and b'nvm2c.c:2109' in order
    assert b'UNIFY a=1662 b=1688' in exact and b'NVM_SHAPE_BOOL' in exact
    assert b'nvm2c.c:1908' in exact
    original = files['source/src/nanoisa/nvm2c.c'].decode()
    before = '(b->shape_opcode == OP_LOAD_LOCAL &&\n         (slot.kind =='
    after = '((b->shape_opcode == OP_LOAD_LOCAL || b->shape_opcode == OP_DUP || b->shape_opcode == OP_PICK) &&\n         (slot.kind =='
    assert original.count(before) == 1 and after not in original
    expected_patch = ''.join(difflib.unified_diff(original.splitlines(True), original.replace(before, after).splitlines(True),
                                                fromfile='a/src/nanoisa/nvm2c.c', tofile='b/src/nanoisa/nvm2c.c'))
    assert (HERE/'proposed-scalar-copy.patch').read_text() == expected_patch
    assert files['observer/proposed-scalar-copy.patch'].decode() == expected_patch
    assert (HERE/'reproduce.nano').read_bytes() == files['products/regression/called0-slice0-write1-guard0-and.nano']
    proposal = data('observer/proposal.json')
    assert proposal['status'] == 'UNAPPLIED_UNQUALIFIED_CALLER_PROPOSAL'
    assert proposal['source_sha256'] == sha(original.encode())
    print(json.dumps(dict(status='LOCALIZATION_INTEGRITY_PASS_PRODUCTION_REGRESSION_STILL_RED',
        source_pin=manifest['source_pin'], records=len(files), sources=len(source_paths),
        graph_checks=312, diagnostic_sources=64, regression_sources=64,
        valid_source_refusals=8, valid_pick_refusals=1, invalid_pick_refused=True,
        caller_patch_applied=False, caller_candidate_compiled=False,
        frame_five_route_acceptance=False, whole_function_cfg=False, release_acceptance=False), indent=2))


if __name__ == '__main__':
    main()
