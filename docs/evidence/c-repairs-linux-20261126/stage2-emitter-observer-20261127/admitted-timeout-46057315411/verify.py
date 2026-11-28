"""I verify retained Stage2 diagnostic bytes without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='0bf733633017fdc0a3fa58e0d7656a591a98f2eeaeb91dc7bf29c9f7cb2d5f2c'
manifest=json.loads(raw)
assert not manifest['qualification'] and not manifest['whole_release']
with tempfile.TemporaryDirectory(prefix='namespace-admission-reader-', dir=args.workspace) as temporary:
    root = Path(temporary)
    used, seen = set(), set()
    for record in manifest['files']:
        relative = Path(record['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert str(relative) not in seen
        seen.add(str(relative))
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with target.open('wb') as stream:
            for key in record['chunks']:
                assert re.fullmatch('[a-f0-9]{64}', key)
                packed = (args.archive / 'objects' / (key + '.gz')).read_bytes()
                metadata = manifest['chunks'][key]
                assert sha(packed) == metadata['gzip_sha256']
                block = gzip.decompress(packed)
                assert sha(block) == key and len(block) == metadata['bytes']
                digest.update(block)
                size += len(block)
                stream.write(block)
                used.add(key)
        assert size == record['size'] and digest.hexdigest() == record['sha256']
    assert used == set(manifest['chunks'])
    job=json.loads((root/'job.json').read_text())
    assert (job['id'],job['run_id'],job['head_sha'],job['conclusion'])==(140702830952,46057315411,'5661e07adeae7007c91cb5e9b17ee9a292bd0fa7','success')
    assert sha((root/'diagnostic.zip').read_bytes())=='9f773284b7ed69e499ab760ed31b59589ec3cb879886734002a7a86e3efed5ab'
    with zipfile.ZipFile(root/'diagnostic.zip') as z:
        assert len(z.namelist())==1813 and z.testzip() is None
        read=lambda name:z.read('stage2-emitter-diagnostic/'+name)
        load=lambda name:json.loads(read(name))
        d=load('diagnostic.json');p=d['product']
        assert all(d[k] for k in ['observation_valid','guards_match','namespace_recovery_complete','native_compile_and_run_control_passed'])
        assert not d['qualification'] and not d['whole_release'] and d['controls_passed']==3
        assert p==load('command/result.json')
        assert p['terminal'] and p['command_timeout'] and p['exit_code']==-9 and p['deadline_seconds']==300
        assert 300<=p['elapsed_seconds']<310 and p['output_complete'] and not p['observer_errors']
        assert sha(read('command/stdout'))==p['stdout_sha256'] and sha(read('command/stderr'))==p['stderr_sha256']
        journal=load('cache-current-restoration.json')
        assert journal['complete'] and journal['entries'] and not journal['qualification']
        assert not journal['original_link_metadata_available']
        for row in journal['entries']:
            assert row['files']
            for name,info in row['files'].items():
                data=read('flattened-cache-currents/'+row['current']+'/'+name)
                assert len(data)==info['size'] and sha(data)==info['sha256']
        for phase in ['compile','execute']:
            c=load('native-control/'+phase+'/result.json')
            assert c['terminal'] and c['exit_code']==0 and not c['command_timeout']
            assert c['output_complete'] and not c['observer_errors']
            assert sha(read('native-control/'+phase+'/stdout'))==c['stdout_sha256']
            assert sha(read('native-control/'+phase+'/stderr'))==c['stderr_sha256']
        assert load('identities-before.json')==load('identities-after-control.json')==load('identities-after.json')
        before=load('observer-sources-before.json')
        assert before==load('observer-sources-after.json')
        for name,expected in before.items(): assert sha(read('controller-sources/'+Path(name).name))==expected
        guard=load('product-tree-guard.json')
        assert guard['matches'] and guard['unchanged_product_entries']==4712
        assert guard['original_product_revision']=='d18fed86140b6a4cf983898ac32a079002f6b784'
        samples=[json.loads(line) for line in read('command/processes.jsonl').splitlines()]
        assert len(samples)==297 and all(a['elapsed_seconds']<b['elapsed_seconds'] for a,b in zip(samples,samples[1:]))
        last=samples[-1]
        assert 299<last['elapsed_seconds']<300
        assert any(row['command']=='nvm2c' and row['cpu']=='00:00:23' for row in last['processes'])
        assert any(row['command']=='nanoc_stage2' and row['cpu']=='00:04:24' for row in last['processes'])
print('I verified my retained admitted timeout, native readiness, namespace journal and byte guards; qualification remains failed.')
