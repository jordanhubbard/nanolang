#!/usr/bin/env python3
"""I inspect retained source-metadata data without executing a native target."""
from pathlib import Path
import gzip
import hashlib
import json
import types

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / 'manifest.json').read_text())
files = {}
for entry in manifest['files']:
    assert entry['path'] not in files
    packed = (HERE / entry['artifact']).read_bytes()
    assert (len(packed), hashlib.sha256(packed).hexdigest()) == (entry['compressed_bytes'], entry['compressed_sha256'])
    raw = gzip.decompress(packed)
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == (entry['bytes'], entry['sha256'])
    files[entry['path']] = raw
report = json.loads(files['native/result.json'])
run = json.loads(files['native/run.json'])
jobs = json.loads(files['native/jobs.json'])
wire = files['native/source-metadata.json']
assert run['id'] == manifest['run'] == 45123800085
assert run['head_sha'] == report['commit'] == manifest['source_commit'] == '0147a23f4b5c8aece857646674603643ee871478'
assert run['status'] == 'completed' and run['conclusion'] == 'success'
assert jobs['total_count'] == 1 and jobs['jobs'][0]['conclusion'] == 'success'
assert all(report[key] is True for key in ('lossless_round_trip', 'expected_large_format_selected', 'all_source_metadata_unchanged', 'git_unchanged', 'success'))
assert report['wire_observations'] == [{'version': 2, 'bytes': 87810293}, {'version': 3, 'bytes': 30687179}]
assert report['wire_bound'] == 32 * 1024 * 1024
assert len(wire) == report['wire_bytes'] == 30687179 <= report['wire_bound']
assert hashlib.sha256(wire).hexdigest() == report['wire_sha256'] == '3fe7b975396e2fe2d09cd18c6a59e54d9be14dc6000691c45d4dd33fb61eb6a9'
assert b'Ran 21 tests' in files['native/controls.log'] and files['native/controls.log'].rstrip().endswith(b'OK')
# I authenticate the recorded Python bytes before loading them.
assert hashlib.sha256(files['source/ci_bootstrap_checkpoint.py']).hexdigest() == 'b7ca7bf0b06247258c4db72fa8a3a18f8866019049f9e3766322aaa28ec194de'
# I load only the recorded, hash-bound Python codec. No native image runs.
codec = types.ModuleType('checkpoint_data_codec')
codec.__file__ = str(HERE / 'recorded_codec.py')
exec(compile(files['source/ci_bootstrap_checkpoint.py'], codec.__file__, 'exec'), codec.__dict__)
decoded = codec.decode_manifest(wire)
assert decoded['version'] == 1 and decoded['commit'] == report['commit']
assert len(decoded['tracked']) == report['tracked_files'] == 357103
assert len(decoded['files']) == report['all_records'] == 357195
assert len(decoded['files']) - len(decoded['tracked']) == 92
assert all(decoded['files'][name]['sha256'] == value for name, value in decoded['tracked'].items())
assert hashlib.sha256(files['source/ci_bootstrap_checkpoint.py']).hexdigest() == decoded['tracked']['scripts/ci_bootstrap_checkpoint.py']
assert codec.encode_manifest(decoded) == wire
result = {'source_commit': report['commit'], 'records': len(files), 'tracked_files': len(decoded['tracked']), 'source_records': len(decoded['files']), 'legacy_v2_bytes': 87810293, 'compact_v3_bytes': len(wire), 'original_bound': report['wire_bound'], 'round_trip_identical_wire': True, 'native_source_guards_reported_unchanged': True, 'scope': 'I validate recorded metadata, terminal identities and hash-bound Python decoding only. I do not execute native products, rerun bootstrap/transfer/consumers, or independently hydrate and hash every Git blob.'}
print(json.dumps(result, indent=2))
