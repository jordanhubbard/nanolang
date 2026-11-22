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

assert manifest['missing'][0]['path']=='tests/nanovm/test_vm'
controller=data('controller/controller.json')
assert controller['status']=='passed'
assert [x['label'] for x in controller['steps']]==['fresh-tools-vm','original-bootstrap3']
assert all(x['exit']==0 for x in controller['steps'])
b=controller['bootstrap_receipt']
assert b==data('images/bin/nanoc_bootstrap.json')
assert b['stage_timeout_seconds']==1800 and b['shadow_timeout_seconds']==30
labels=['seed','seed-verify','stage1','stage1-verify','stage1-translate','stage1-native','stage1-hello-compile','stage1-hello-execute','stage2','stage2-verify','stage2-translate','stage2-native','stage2-hello-compile','stage2-hello-execute','seed-final-verify','stage1-final-verify','stage2-final-verify']
assert [x['label'] for x in b['steps']]==labels
assert all(x['exit_code']==0 for x in b['steps'])
assert b['phase']=='stage2' and b['raw_stage1_stage2_equal']
for x in b['steps']:
    assert x['elapsed_seconds']<1800
    assert entries['bootstrap/'+Path(b['work']).name+'/'+Path(x['log']).name]['sha256']==x['log_sha256']
for label,producer in [('stage1','nanoc_seed.nvm'),('stage2','nanoc_stage1.nvm')]:
    step=next(x for x in b['steps'] if x['label']==label)
    assert Path(step['argv'][0]).name=='nano_vm' and Path(step['argv'][1]).name==producer
assert entries['images/bin/nanoc_stage1.nvm']['sha256']==entries['images/bin/nanoc_stage2.nvm']['sha256']=='d63819e232b88781a703c68c10bff9ac79758b64d9ddda2bacb2da51c7421a91'
assert entries['images/bin/nanoc_stage1.nvm']['bytes']==entries['images/bin/nanoc_stage2.nvm']['bytes']==1039116
for name,digest in b['artifacts'].items():assert entries['images/bin/'+name]['sha256']==digest
for name,digest in b['tools'].items():assert entries['images/bin/'+name]['sha256']==digest
source_map={x['path']:x['sha256'] for x in manifest['git_sources']}
source_map['modules/index.json']=entries['generated-sources/modules/index.json']['sha256']
assert len(b['sources'])==1340 and all(source_map[name]==digest for name,digest in b['sources'].items())
for name,digest in b['hosts'].items():
    relative=name.split('/generic-array-raw-linux-20261122/',1)[1]
    assert entries['hosts/'+relative]['sha256']==digest
assert manifest['symlinks']=={'bin/nanoc':'nanoc_stage2'}
log=raw('controller/original-bootstrap3.log').decode()
assert 'bin/nanoc succeeded with bin/nanoc_c removed' in log
assert '569591 passed, 0 failed, 569591 total' in raw('controller/fresh-tools-vm.log').decode()
cutoff=json.loads((HERE/'post-bootstrap-cutoff.json').read_text())
assert [x['label'] for x in cutoff['steps']]==['original-contexts-gcc','original-contexts-clang']
assert all(x['exit']==0 for x in cutoff['steps'])
for name in ('gcc','clang'):
    terminal=raw('post-bootstrap/original-contexts-'+name+'.log').decode()
    assert 'Ran 4 tests' in terminal and terminal.rstrip().endswith('OK')
inputs=[data(path) for path in entries if path.startswith('original-contexts/') and path.endswith('/inputs.json')]
assert len(inputs)==16
assert {Path(x['provider']).name for x in inputs}=={'nanoc_c','nano_virt','nanoc_stage1','nanoc_stage2'}
assert all(x['source_before']==x['source_after'] and x['provider_before']==x['provider_after'] for x in inputs)
for row in inputs:
    name='bin/'+Path(row['provider']).name
    assert entries['images/'+name]['sha256']==row['provider_before']
    assert source_map['tests/unit/'+Path(row['source']).name]==row['source_before']
tree={}
for line in subprocess.check_output(['git','ls-tree','-r',manifest['source_commit']],cwd=REPO,text=True).splitlines():
    fields,path=line.split('\t',1);tree[path]=fields.split()[2]
assert all(tree[x['path']]==x['git_blob'] for x in manifest['git_sources'])
counts=manifest['counts']
assert counts==dict(files=len(entries),git_sources=len(manifest['git_sources']),raw_bytes=sum(x['bytes'] for x in entries.values()),unique_objects=len(objects),compressed_bytes=sum(x[0] for x in objects.values()))
print(json.dumps(dict(status='passed',logical_records=len(entries)+len(manifest['git_sources']),**counts,bootstrap_input_count=1340,raw_module_bytes=1039116,raw_module_sha256='d63819e232b88781a703c68c10bff9ac79758b64d9ddda2bacb2da51c7421a91',original_context_methods_per_compiler=4,source_providers=4,paired_darwin_claimed=False,later_source_claimed=False,full_route_audit_claimed=False),indent=2))
