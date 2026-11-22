"""I independently check my retained development bytes and finite gate results."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
manifest = json.loads((HERE / 'manifest.json').read_text())
entries = {entry['path']: entry for entry in manifest['files']}
assert len(entries) == len(manifest['files'])
objects = {}
for entry in entries.values():
    artifact = entry['artifact']
    if artifact not in objects:
        path = (HERE / artifact).resolve(strict=True)
        assert path.is_relative_to(HERE / 'objects')
        packed = path.read_bytes()
        raw = gzip.decompress(packed)
        objects[artifact] = (len(packed), hashlib.sha256(packed).hexdigest(),
                             len(raw), hashlib.sha256(raw).hexdigest())
    packed_size, packed_sha, raw_size, raw_sha = objects[artifact]
    assert (packed_size, packed_sha, raw_size, raw_sha) == (
        entry['compressed_bytes'], entry['compressed_sha256'], entry['bytes'], entry['sha256']), entry['path']
assert {path.name for path in (HERE / 'objects').iterdir()} == {Path(path).name for path in objects}
process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO,
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for entry in manifest['git_sources']:
    blob = entry['git_blob']
    assert re.fullmatch('[0-9a-f]{40}', blob)
    process.stdin.write((blob + '\n').encode())
    process.stdin.flush()
    identity, kind, size = process.stdout.readline().decode().split()
    assert identity == blob and kind == 'blob'
    raw = process.stdout.read(int(size))
    assert process.stdout.read(1) == b'\n'
    assert (len(raw), hashlib.sha256(raw).hexdigest()) == (entry['bytes'], entry['sha256']), entry['path']
process.stdin.close()
assert process.wait(timeout=30) == 0

def raw(path):
    return gzip.decompress((HERE / entries[path]['artifact']).read_bytes())

def data(path):
    return json.loads(raw(path))

for name, expected in [('mutation', 10), ('neighbors', 25)]:
    result = data('retained/native-acceptance/' + name + '/result.json')
    assert result['discovered'] == result['ran'] == expected
    assert result['successful'] and not result['failures'] and not result['errors'] and not result['skipped']
qualified = data('retained/native-acceptance/manifest.json')
assert qualified['status'] == 'passed' and qualified['guards_unchanged']
assert {item['path']: item['sha256'] for item in manifest['git_sources']} == qualified['source_hashes']
for name, digest in qualified['images'].items():
    assert entries['images/' + name]['sha256'] == digest
native = data('observer/native-final.json')
assert native['status'] == 'passed'
assert native['sha256'] == qualified['images']['bin/nanoc_receiver-flow-final']
labels = [step['label'] for step in native['bootstrap_manifest']['steps']]
assert labels == ['seed', 'seed-verify', 'receiver-flow-final-translate', 'receiver-flow-final-native',
                  'receiver-flow-final-hello-compile', 'receiver-flow-final-hello-execute']
assert all(step['exit_code'] == 0 for step in native['bootstrap_manifest']['steps'])
assert native['bootstrap_manifest']['shadow_timeout_seconds'] == 30
assert native['bootstrap_manifest']['stage_timeout_seconds'] == 1800
for path, digest in native['bootstrap_manifest']['hosts'].items():
    relative = path.split('/seed-final/module-cache/', 1)[1]
    assert entries['retained/seed-final/module-cache/' + relative]['sha256'] == digest
mutation = data('retained/native-acceptance/mutation/commands.json')
assert len(mutation) == 483 and not any('error' in command for command in mutation)
assert {command['timeout'] for command in mutation} == {180, 30}
mutation_sources = {item['sha256'] for command in mutation for item in command['inputs']
                    if item['path'].endswith('/main.nano')}
assert len(mutation_sources) == 101
preview = data('preflight/make-preview/commands.json')
assert mutation_sources == {item['sha256'] for command in preview for item in command['inputs']
                            if item['path'].endswith('/main.nano')}
sanitizers = data('sanitizers/checked-result.json')
assert sanitizers['discovered'] == sanitizers['ran'] == 10
assert sanitizers['successful'] and not sanitizers['failures'] and not sanitizers['errors'] and not sanitizers['skipped']
scanned = data('retained/compiler-sanitizers/checked-diagnostics/commands.json')
assert len(scanned) == 161 and not any(command['sanitizer_diagnostics'] for command in scanned)
assert mutation_sources == {item['sha256'] for command in scanned for item in command.get('inputs', [])}
assert '569591 passed, 0 failed, 569591 total' in raw('observer/nanovm.log').decode()
assert data('observer/frozen-linux-guard.json')['status'] == 'passed'
counts = manifest['counts']
assert counts == dict(files=len(entries), git_sources=len(manifest['git_sources']),
    raw_bytes=sum(entry['bytes'] for entry in entries.values()), unique_objects=len(objects),
    compressed_bytes=sum(value[0] for value in objects.values()))
print(json.dumps(dict(status='passed',logical_records=len(entries)+len(manifest['git_sources']),
    **counts,mutation_methods=10,mutation_sources=101,mutation_commands=483,neighbor_methods=25,
    sanitizer_methods=10,sanitizer_scanned_commands=161,sanitizer_reports=0,
    raw_self_compilation_claimed=False,paired_darwin_claimed=False),indent=2))
