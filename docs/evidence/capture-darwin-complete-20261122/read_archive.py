#!/usr/bin/env python3
"""I inspect original Darwin capture evidence without executing a native target."""
from pathlib import Path
import gzip
import hashlib
import json
import struct
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
manifest = json.loads((HERE / 'manifest.json').read_text())
files = {}
cache = {}
for entry in manifest['files']:
    assert entry['path'] not in files
    artifact = entry['artifact']
    if artifact not in cache:
        packed = (HERE / artifact).read_bytes()
        cache[artifact] = (packed, gzip.decompress(packed))
    packed, raw = cache[artifact]
    assert (len(packed), hashlib.sha256(packed).hexdigest()) == (entry['compressed_bytes'], entry['compressed_sha256'])
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == (entry['bytes'], entry['sha256'])
    files[entry['path']] = raw

def data(name):
    return json.loads(files[name])

report = data('native-observation/original-darwin-result.json')
assert report['pin'] == '0d394f47e9cc5deb3d5eab4581f4605f98f6adf8'
assert report['discovered'] == report['ran'] == 129
assert report['failures'] == report['errors'] == 0
assert report['success'] and report['all_source_hashes_unchanged'] and report['native_images_unchanged']
previous = data('previous-refusal/original-darwin-result.json')
assert report['skipped'] == previous['skipped'] and len(report['skipped']) == 19
assert previous['failures'] == previous['errors'] == 1
before = data('native-observation/native-before.json')
assert before == data('native-observation/native-after.json')
probe = files['native-observation/test_module_generation_probe']
assert probe[:4] == b'\xcf\xfa\xed\xfe' and struct.unpack_from('<I', probe, 4)[0] == 0x100000c
probe_name = next(name for name in before if name.endswith('/obj/test_module_generation_probe'))
assert hashlib.sha256(probe).hexdigest() == before[probe_name] == '1067b1f76c0471244d0a5c15de38b5c1513e220d4ae2c0f0a10a1ef74dc1c433'
assert files['native-observation/controller-sha256.txt'].decode().strip() == hashlib.sha256(files['native-observation/run-original-darwin.py']).hexdigest()
command = data('native-observation/build-command.json')
assert command == ['make', '-f', 'Makefile.gnu', '-j1', 'obj/test_module_generation_probe', 'CC=/usr/bin/cc',
    'CFLAGS=-Wall -Wextra -Werror -std=c99 -O2 -g -fPIC -Isrc -D_GNU_SOURCE -I/opt/homebrew/opt/openssl@3/include',
    'LDFLAGS=-lm -L/opt/homebrew/opt/openssl@3/lib -lcrypto']
assert data('native-observation/build-result.json') == {'returncode': 0, 'source_pin': report['pin']}
inputs = data('native-observation/source-inputs.json')
assert len(inputs) == report['inputs']
process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for name, expected in inputs.items():
    raw = files['executed-source/' + name]
    assert hashlib.sha256(raw).hexdigest() == expected
    process.stdin.write((report['pin'] + ':' + name + '\n').encode())
    process.stdin.flush()
    header = process.stdout.readline().decode().split()
    assert len(header) == 3 and header[1] == 'blob', name
    source = process.stdout.read(int(header[2]))
    assert process.stdout.read(1) == b'\n' and source == raw, name
process.stdin.close()
assert process.wait(timeout=30) == 0
matrix = data('native-observation/response-driver-grammar-matrix.json')
assert len(matrix) == 600
assert all(row['captured_returncode'] == 0 and (row['native_returncode'], row['native_stdout']) ==
    (row['replay_returncode'], row['replay_stdout']) for row in matrix)
assert {row['quote'] for row in matrix} == {'', "'", '"'}
assert len({row['compiler'] for row in matrix}) == 2
log = files['native-observation/original-darwin-tests.log']
assert b'Ran 129 tests' in log and log.rstrip().endswith(b'OK (skipped=19)')
assert b'test_response_quoted_backslashes_follow_selected_driver' in log
print(json.dumps({'records': len(files), 'Git_source_inputs': len(inputs), 'all_hashes_and_lengths_match': True,
    'original_methods': 129, 'unchanged_conditional_skips': 19, 'response_cases': 600,
    'response_returncode_stdout_matches': 600,
    'response_stderr_matches': sum(row['native_stderr'] == row['replay_stderr'] for row in matrix),
    'source_and_observed_tool_images_unchanged': True,
    'scope': 'I validate Python/Git data only. I execute no native target. This capture component at0d is separate from later-source/full-platform/release acceptance; installed system tools and dynamic-library closure are not redistributed or claimed frozen.'}, indent=2))
