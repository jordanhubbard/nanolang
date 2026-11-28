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

for name, expected in [('seed-acceptance',7),('receiver',12),('neighbors',25)]:
    result = data('development/'+name+'/observation/result.json')
    assert result['discovered'] == result['ran'] == expected
    assert result['successful'] and not result['failures'] and not result['errors'] and not result['skipped']
    commands = data('development/'+name+'/observation/commands.json')
    assert not any('error' in command for command in commands)
    assert {command['timeout'] for command in commands} == ({30,90,120,180} if name=='neighbors' else {180,30})
seed = data('development/compiler-seed/manifest.json')
assert seed['status'] == 'passed'
b = seed['bootstrap_manifest']
assert [s['label'] for s in b['steps']] == ['seed','seed-verify']
assert all(s['exit_code']==0 for s in b['steps'])
assert b['stage_timeout_seconds']==1800 and b['shadow_timeout_seconds']==30
source_map = {entry['path']: entry['sha256'] for entry in manifest['git_sources']}
assert all(source_map[name]==digest for name,digest in b['sources'].items())
for path, digest in b['tools'].items():
    relative = 'bin/'+path
    assert entries['images/'+relative]['sha256']==digest
for path, digest in b['hosts'].items():
    relative = path.split('/compiler-seed/module-cache/',1)[1]
    assert entries['development/compiler-seed/module-cache/'+relative]['sha256']==digest
sanitizers = data('development/compiler-sanitizers/result.json')
assert sanitizers['discovered']==sanitizers['ran']==19
assert sanitizers['successful'] and not sanitizers['failures'] and not sanitizers['errors'] and not sanitizers['skipped']
scanned = data('development/compiler-sanitizers/diagnostics/commands.json')
assert len(scanned)==478 and {command['timeout'] for command in scanned}=={180,30}
assert not any(command['sanitizer_diagnostics'] for command in scanned)
for name,digest in sanitizers['images'].items(): assert entries['images/'+name]['sha256']==digest
original = data('development/original-contexts.json')
assert [item['name'] for item in original]==['gcc','clang'] and all(item['exit']==0 for item in original)
for name in ('gcc','clang'): assert 'Ran 4 tests' in raw('development/original-'+name+'.log').decode()
assert '569591 passed, 0 failed, 569591 total' in raw('development/vm-identifiers.log').decode()
assert '0 reserved names' in raw('development/vm-identifiers.log').decode()
review = data('selector-review/native-supported-result.json')
assert review['settings_unchanged'] and not review.get('environment_changed',False)
assert review['compiler_before']==review['compiler_after']
assert 'Ran 8 tests' in raw('selector-review/controls.log').decode()
search = data('selector-review/native-search.json')
assert len(search)==4 and sum(item['selected'] for item in search)==2
counts=manifest['counts']
assert counts==dict(files=len(entries),git_sources=len(manifest['git_sources']),raw_bytes=sum(entry['bytes'] for entry in entries.values()),unique_objects=len(objects),compressed_bytes=sum(value[0] for value in objects.values()))
cohorts={}
for name in ('seed-acceptance','receiver','neighbors'):
    commands=data('development/'+name+'/observation/commands.json')
    sources={item['sha256'] for command in commands for item in command['inputs'] if item['path'].endswith('/main.nano')}
    cohorts[name]=dict(commands=len(commands),main_source_bytes=len(sources))
    assert len(commands)=={'seed-acceptance':102,'receiver':615,'neighbors':225}[name]
    assert len(sources)=={'seed-acceptance':18,'receiver':143,'neighbors':9}[name]
print(json.dumps(dict(status='passed',logical_records=len(entries)+len(manifest['git_sources']),**counts,cohorts=cohorts,sanitizer_scanned_commands=len(scanned),raw_bootstrap_claimed=False,paired_darwin_claimed=False),indent=2))
