#!/usr/bin/env python3
"""I verify my one-line scalar-copy repair and complete five-route evidence."""
from pathlib import Path
import gzip
import hashlib
import json
import re
import subprocess
import struct
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    assert manifest['source_pin'].startswith('0962d3c9db')
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
    profile = json.loads((HERE/'tool-profile.json').read_text())
    assert len(profile['tools']) == 2
    assert all(row['exit'] == 0 and len(row['sha256']) == 64 for row in profile['tools'])
    assert '13.3.0' in profile['tools'][0]['stdout']
    assert '2.42' in profile['tools'][1]['stdout']
    q = json.loads(files['journal/qualification.json'])
    assert q['source_pin'] == manifest['source_pin']
    assert q['grant'].endswith('issuecomment-7093099516')
    assert q['one_line_only'] is True and q['wider_merge_relaxation'] is False
    assert q['build_budget'] == 600
    before, after = q['before']['files'], q['after']
    original = files['controls/original-consumer-source']
    current = files['source/src/nanoisa/nvm2c.c']
    first = b'        (b->shape_opcode == OP_LOAD_LOCAL &&\n'
    second = b'        ((b->shape_opcode == OP_LOAD_LOCAL || b->shape_opcode == OP_DUP || b->shape_opcode == OP_PICK) &&\n'
    assert original.count(first) == 1 and original.replace(first, second) == current
    assert sha(original) == '12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67'
    assert sha(current) == '994d83e225343cfee19f5674cc30259fcea6bc3db44d65911764fd63489f31c0'
    assert sha(files['controls/original-consumer']) == 'c94adddd92eefc081ee5cb28df971a36fe2f7cbe8e9730ec68ca6b1c3d24a7ea'
    assert sha(files['images/bin/nvm2c']) == q['activation']['sha256'] == '007dab36cbffea6c151072155d99d220c2c1aaa3b2ea1847932bba5b00b93d8c'
    assert sha(files['images/bin/nanoc_stage2']) == '346dc0d73af5960dea863fa724a0b1845f5613dff10b8661180ed74bf4eb3d83'
    for name, row in after.items():
        raw = files[('images/' if name.startswith('bin/') else 'source/')+name]
        assert (len(raw), sha(raw)) == (row['bytes'], row['sha256'])
        if name not in ('src/nanoisa/nvm2c.c', 'bin/nvm2c'):
            assert row == before[name]
    changes = subprocess.check_output(['git','diff','--numstat',manifest['source_pin']+'^',manifest['source_pin'],'--','src','src_nano','tests','Makefile.gnu'],cwd=ROOT).decode().splitlines()
    assert changes == ['1\t1\tsrc/nanoisa/nvm2c.c']
    commands = json.loads(files['journal/commands.json'])
    assert commands['result'] == q['tests']
    assert q['tests']['success'] and q['tests']['tests'] == 2
    assert (q['tests']['failures'], q['tests']['errors'], q['tests']['skipped']) == (0, 0, 0)
    assert len(commands['commands']) == 898
    failures = []
    for index, row in enumerate(commands['commands']):
        assert row['state'] == 'terminal'
        stdout, stderr = files[f'journal/{index:04d}.stdout'], files[f'journal/{index:04d}.stderr']
        assert sha(stdout) == row['stdout_sha256'] and sha(stderr) == row['stderr_sha256']
        if row['exit']:
            assert row['exit'] == 1
            assert any(Path(arg).name in ('pick-1.nvm','pick-1.nasm') for arg in row['command'])
            failures.append(row)
    assert len(failures) == 3
    controls = json.loads(files['journal/controls.json'])
    assert len(controls) == 9 and all(row['exit'] == 0 for row in controls)
    for prefix in ('scalar-copy', 'scalar-copy-asan'):
        assert b'312 passed, 0 failed' in files['journal/'+prefix+'-run.stdout']
    for prefix in ('original-shape', 'original-shape-asan'):
        assert b'3890 passed, 0 failed' in files['journal/'+prefix+'-run.stdout']
    matched = json.loads((HERE/'matched-baseline.json').read_text())
    assert len(matched['rows']) == 2
    for row in matched['rows']:
        assert row['consumer_sha256'] == sha(files['controls/original-consumer'])
        assert row['module_sha256'] == sha(files['products/scalar/'+row['module']])
        assert row['exit'] == 1 and row['before_hex'] == row['after_hex']
        assert bytes.fromhex(row['after_hex']) == b'I retain my prior C output.\0\xff\n'
        assert 'exactly constrained bool destination' in row['stderr']
    scalar = json.loads(files['products/scalar/manifest.json'])
    assert len(scalar['cases']) == 64 and len(scalar['commands']) == 134
    assert sum(row['exit'] != 0 for row in scalar['commands']) == 3
    for row in scalar['cases']:
        stem = row['name']
        assert row['aot_exit'] == 0 and not row['prior_preserved']
        assert sha(files['products/scalar/'+stem+'.nano']) == row['source_sha256']
        assert sha(files['products/scalar/'+stem+'.nvm']) == row['module_sha256']
        assert sha(files['products/scalar/'+stem+'.c']) == row['output_sha256']
    raw0 = files['products/scalar/pick-0.nvm']
    raw1 = files['products/scalar/pick-1.nvm']
    mutation = scalar['negative_mutation']
    assert raw0[:4] == raw1[:4] == b'NVM\x02'
    offset = mutation['byte_offset']
    assert raw0[offset-1:offset+2] == b'\xf4\0\0'
    assert raw1[offset-1:offset+2] == b'\xf4\1\0'
    expected = bytearray(raw0)
    struct.pack_into('<H', expected, offset, 1)
    struct.pack_into('<I', expected, 36, zlib.crc32(expected[40:]) & 0xffffffff)
    assert raw1 == bytes(expected) and sha(raw1) == mutation['sha256']
    assert files['products/scalar/pick-1.c'] == b'I retain my prior C output.\0\xff\n'
    labels = {row['label']: row for row in scalar['commands']}
    for label in ('pick-0-assembler','pick-0-verify','pick-0-aot'):
        assert labels[label]['exit'] == 0
    for label in ('pick-1-assembler','pick-1-verify','pick-1-aot'):
        assert labels[label]['exit'] == 1
    frame = json.loads(files['products/frame/manifest.json'])
    assert (frame['flow_operations'],frame['normalized_flow_states'],frame['queue_traces'],frame['frame_cases']) == (168386,289,7,665)
    assert len(frame['corpus']) == 1001
    labels = ['actual-c','c-seed-native','c-producer-nanoisa-vm','c-producer-sanitized-native','nanolang-producer-nanoisa-vm','nanolang-producer-sanitized-native']
    assert [row['label'] for row in frame['routes']] == labels
    expected_reports = files['products/frame/actual-c.reports']
    assert sha(expected_reports) == 'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0'
    reports = expected_reports.decode().splitlines()
    assert len(reports) == 1001
    for index, row in enumerate(frame['corpus']):
        path = Path(row['path'])
        assert path.parent == Path(q['product_roots']['frame'])
        raw = files['products/frame/'+path.name]
        assert (len(raw),sha(raw)) == (row['bytes'],row['sha256'])
        if row['name'].startswith(('global-algebra-', 'bounded-fifo-')):
            assert reports[index] == row['expected']
        else:
            assert reports[index].startswith(row['expected'])
    for row in frame['routes']:
        assert row['reports'] == 1001
        raw = files['products/frame/'+row['label']+'.reports']
        assert raw == expected_reports and sha(raw) == row['sha256']
    for name in ('decoder.nvm','source-decoder.nvm'):
        assert files['products/frame/'+name][:4] == b'NVM\x02'
    for name in ('decoder-native','decoder-aot','source-decoder-aot'):
        raw = files['products/frame/'+name]
        assert raw[:6] == b'\x7fELF\x02\x01' and struct.unpack_from('<H',raw,18)[0] == 183
    assert len(q['provider_identities']) > 60
    print(f"PASS: {len(files)} records/{len(paths)} Git inputs; exact one-line repair; 1001 carriers identical in actual C/all five routes; strict negatives retained.")


if __name__ == '__main__':
    main()
