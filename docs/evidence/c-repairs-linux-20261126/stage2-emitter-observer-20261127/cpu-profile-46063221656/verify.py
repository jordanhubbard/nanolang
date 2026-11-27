"""I verify my CPU sampler controls without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='a772a232f0b769f4c8835f7c20048941bd88c0e41c3b1f93e74564f09d1a4ffe'
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
    import collections,io,zipfile
    job=json.loads((root/'job.json').read_text())
    assert (job['id'],job['run_id'],job['head_sha'],job['conclusion'])==(140722238009,46063221656,'9913677caed93a279563ee719fcb5a55ef2b1f03','success')
    assert sha((root/'diagnostic.zip').read_bytes())=='052b92c18ba81616bfed8594c067498f3477ada193f75602e97e491596b16981'
    with zipfile.ZipFile(root/'diagnostic.zip') as z:
        assert len(z.namelist())==1840 and z.testzip() is None
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
        assert journal['complete'] and len(journal['entries'])==9 and not journal['qualification']
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
        ids=load('identities-before.json')
        assert ids==load('identities-after-control.json')==load('identities-after.json') and len(ids)==1406
        sources=load('observer-sources-before.json')
        assert sources==load('observer-sources-after.json') and len(sources)==5
        for name,expected in sources.items():assert sha(read('controller-sources/'+Path(name).name))==expected
        guard=load('product-tree-guard.json')
        assert guard['matches'] and guard['unchanged_product_entries']==4712
        assert guard['original_product_revision']=='d18fed86140b6a4cf983898ac32a079002f6b784'
        assert load('invocation.json')['observer_revision']==job['head_sha']
        original=read('original-artifact.zip')
        assert sha(original)=='f80edbf52d706501319cdefa9d5dcdce3ea933b5b3723f2621cf1fffaa258dc5'
        with zipfile.ZipFile(io.BytesIO(original)) as nested:
            assert len(nested.namelist())==4066 and nested.testzip() is None
            receipt=json.loads(nested.read(next(name for name in nested.namelist() if name.endswith('/bin/nanoc_bootstrap.json'))))
            assert receipt['raw_stage1_stage2_equal'] and len(receipt['steps'])==17
            assert all(step['exit_code']==0 for step in receipt['steps'])
            assert receipt['stage_timeout_seconds']==1800 and receipt['shadow_timeout_seconds']==30
            assert len(receipt['hosts'])==6
            for name,expected in receipt['hosts'].items():
                relative=str(Path(name).relative_to(receipt['root']))
                assert sha(nested.read('nanolang/nanolang/'+relative))==expected
        samples=[json.loads(line) for line in read('command/processes.jsonl').splitlines()]
        assert len(samples)==298
        assert all(process['group']==p['pid'] for sample in samples for process in sample['processes'])
        assert 'Ran 21 tests in ' in z.read('stage2-emitter-controls/stderr').decode()
        provision=json.loads(z.read('stage2-emitter-controls/perf-provision.json'))
        assert provision['perf_event_paranoid']=='-1' and provision['diagnostic_worker_only']
        c=load('cpu-sampler-control/control.json')
        assert c['valid'] and c['sample_count']==95 and c['before']==c['after']
        for directory,count in [('cpu-sampler-control/sample/',95),('command/cpu-profile/',14341)]:
            record=load(directory+'record.json')
            assert record['valid'] and record['event']=='cpu-clock:u' and record['actual_samples']==count
            assert record['interrupt_sent'] and record['exit_code']==-2 and not record.get('forced_stop')
            assert record['perf_sha256_before']==record['perf_sha256_after']=='c6c3c8573dfafe0592431078a87e88639029c3ec15e3300aae00f52e37547831'
            with z.open('stage2-emitter-diagnostic/'+directory+'perf.data') as stream:
                h=hashlib.sha256();size=0
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);size+=len(block)
            assert h.hexdigest()==record['data_sha256'] and size==record['data_bytes']
            for name in ['report','script','buildids']:
                row=record[name];assert row['exit_code']==0
                assert sha(read(directory+name+'.stdout'))==row['stdout_sha256']
                assert sha(read(directory+name+'.stderr'))==row['stderr_sha256']
        record=load('command/cpu-profile/record.json');assert record==p['cpu_profile']
        script=read('command/cpu-profile/script.stdout').decode()
        analysis=json.loads((root/'caller-analysis.json').read_text())
        assert sha(script.encode())==analysis['script_sha256']
        header=re.compile(r'^(.+?)\s+(\d+)\s+(\d+\.\d+):\s+(\d+)\s+(\S+):\s*$')
        frame=re.compile(r'^\s+[a-fA-F0-9]+\s+(\S+)\s+\(')
        inclusive=collections.Counter();total=0;count=0;period=None;symbols=set()
        for line in script.splitlines()+['END 0 0.0: 0 end:']:
            match=header.match(line)
            if match:
                if period is not None:
                    count+=1;total+=period
                    for symbol in symbols:inclusive[symbol]+=period
                period=int(match[4]);symbols=set()
            elif period is not None:
                match=frame.match(line)
                if match:symbols.add(match[1].split('+')[0])
        assert count==analysis['samples']==14341 and total==analysis['total_period']==292673465583
        for row in analysis['inclusive']:
            assert row['period']==inclusive[row['symbol']]
            assert abs(row['percent']-100*row['period']/total)<1e-9
        assert abs(100*inclusive['nvm_verify_function_max_stack']/total-21.4211)<.0001
        assert abs(100*inclusive['nroot_add']/total-18.8899)<.0001
        assert 'nvm_verify_function_max_stack(mod, i, &depth).ok' in (root/'source/src/nanoisa/nvm_v2_convert.c').read_text()
print('I verified95 calibration and14341 product CPU samples, six retained host DSOs, admission/guards and the original timeout; this is diagnostic evidence, not qualification.')
