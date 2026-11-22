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

for name, expected, command_count in [('accepted-scopes',9,178),('complete-graph',1,91),('final-components',5,1660),('regression',44,942)]:
    result=data('development/'+name+'/observation/result.json')
    assert result['discovered']==result['ran']==expected
    assert result['successful'] and not result['failures'] and not result['errors'] and not result['skipped']
    commands=data('development/'+name+'/observation/commands.json')
    assert len(commands)==command_count and not any('error' in x for x in commands)
seed=data('development/compiler-seed/manifest.json')
assert seed['status']=='passed'
b=seed['bootstrap_manifest']
assert [s['label'] for s in b['steps']]==['seed','seed-verify']
assert all(s['exit_code']==0 for s in b['steps'])
assert b['stage_timeout_seconds']==1800 and b['shadow_timeout_seconds']==30
sources={x['path']:x['sha256'] for x in manifest['git_sources']}
assert all(sources[name]==digest for name,digest in b['sources'].items())
assert data('development/final-seed-guard.json')['status']=='passed'
for name,digest in b['tools'].items():assert entries['images/bin/'+name]['sha256']==digest
assert entries['images/bin/nanoc_seed.nvm']['sha256']==b['artifacts']['nanoc_seed.nvm']
for path,digest in b['hosts'].items():
    relative=path.split('/compiler-seed/module-cache/',1)[1]
    assert entries['development/compiler-seed/module-cache/'+relative]['sha256']==digest
prefix='development/complete-graph/tmp/nano-affine-graph-bdxzcv7b/'
graph=data(prefix+'manifest.json')
assert len(graph['corpus'])==218 and len(graph['routes'])==5
expected=[row['expected'] for row in graph['corpus']]
assert len({row['name'] for row in graph['corpus']})==218
for index,row in enumerate(graph['corpus']):
    assert entries[prefix+f'{index:04d}.bin']['sha256']==row['sha256']
assert raw(prefix+'actual-c.reports').decode().splitlines()==expected
for route in graph['routes']:
    assert route['reports']==218
    assert raw(prefix+route['name']+'.reports').decode().splitlines()==expected
state=data('state-neighbor/manifest.json')
assert state['cases']==3141 and state['independent_expectations']==383 and len(state['routes'])==5
assert all(route['reports']==3141 for route in state['routes'])
actual=raw('state-neighbor/actual-c.reports')
for route in state['routes']:assert raw('state-neighbor/'+route['name']+'.reports')==actual
positive_dirs=sorted({p.rsplit('/',1)[0] for p in entries if p.startswith('development/accepted-scopes/tmp/nano-emission-scope-') and p.endswith('/commands.json')})
assert len(positive_dirs)==10
assert sum(len(data(path+'/commands.json')) for path in positive_dirs)==112
for path in positive_dirs:
    commands=data(path+'/commands.json')
    assert all(row['exit']==0 for row in commands)
    for index in range(len(commands)):
        output=raw(path+'/'+str(index)+'.stdout')+raw(path+'/'+str(index)+'.stderr')
        assert b'I require a function value for a bound call.' not in output
        assert b'Field access requires a struct value.' not in output
san=data('development/scope-asan/result.json')
assert san['ran']==9 and san['successful'] and not san['failures'] and not san['errors'] and not san['skipped'] and not san['sanitizer_reports']
negative=data('development/negative-asan/result.json')
assert negative['ran']==2 and negative['successful'] and negative['commands']==66 and negative['sources']==22 and negative['sanitizer_reports']==0
commands=data('development/negative-asan/commands.json')
assert len(commands)==66 and all(row['exit']!=0 and row['timeout']==180 for row in commands)
for path in entries:
    if ((path.startswith('development/scope-asan/tmp/') or path.startswith('development/negative-asan/')) and path.endswith(('.stdout','.stderr'))):
        output=raw(path)
        assert not any(token in output for token in (b'AddressSanitizer',b'UndefinedBehaviorSanitizer',b'runtime error:')),path
assert '2240 affine bytecode checks passed' in raw('development/original-c-identifiers.log').decode()
assert '3249 affine bytecode checks passed' in raw('development/original-c-identifiers.log').decode()
pin=manifest['production_source_commit']
tree={}
for line in subprocess.check_output(['git','ls-tree','-r',pin],cwd=REPO,text=True).splitlines():
    fields,path=line.split('\t',1);tree[path]=fields.split()[2]
assert all(tree[x['path']]==x['git_blob'] for x in manifest['git_sources'])
counts=manifest['counts']
assert counts==dict(files=len(entries),git_sources=len(manifest['git_sources']),raw_bytes=sum(x['bytes'] for x in entries.values()),unique_objects=len(objects),compressed_bytes=sum(x[0] for x in objects.values()))
print(json.dumps(dict(status='passed',logical_records=len(entries)+len(manifest['git_sources']),**counts,graph_cases=218,graph_routes=5,state_traces=3141,scope_methods=9,regression_methods=44,negative_sanitizer_commands=66,sanitizer_reports=0,affine_cfg_admitted=False,private_profile_admitted=False,paired_darwin_claimed=False),indent=2))
