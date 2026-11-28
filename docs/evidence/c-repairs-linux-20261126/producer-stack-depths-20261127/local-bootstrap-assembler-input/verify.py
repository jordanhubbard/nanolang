"""I verify my failed local bootstrap without executing a product."""
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='571be4add68836fca3aebd118fc440e0975ddd5ee990bae9ad2dad1a3ea3509d'
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
    load=lambda name:json.loads((root/name).read_text())
    d=load('outer/bootstrap/result.json');assert d['qualification'] and not d['whole_release']
    assert d['exit']==2 and not d['command_timeout'] and d['guards_match'] and d['source_before']==d['source_after']
    assert d['argv']==['make','-f','Makefile.gnu','-j1','CC=/usr/bin/cc','bootstrap3']
    manifest=load('bootstrap/manifest.json');assert manifest['phase']=='building-stage1'
    assert manifest['stage_timeout_seconds']==1800 and manifest['shadow_timeout_seconds']==30
    assert [(s['label'],s['exit_code']) for s in manifest['steps']]==[('seed',0),('seed-verify',0),('stage1',0),('stage1-verify',0),('stage1-translate',0),('stage1-native',1)]
    assert all(s['elapsed_seconds']<1800 for s in manifest['steps'])
    native=manifest['steps'][-1];assert native['argv'][0]=='/usr/bin/cc' and '-std=c11' in native['argv'] and '-O1' in native['argv']
    assert (root/'bootstrap/05-stage1-native.log').read_text()=="Assembler messages:\nError: can't open /tmp/ccA4aFXT.s for reading: No such file or directory\n"
    ids=load('outer/post-failure-identities.json')
    for entry in list(ids['tools'].values())+list(ids['artifacts'].values()):assert entry['expected']==entry['actual']
    assert ids['native_c']['bytes']==19631247 and ids['native_c']['sha256']==sha((root/'bootstrap/stage1.c').read_bytes())
    for name,expected in manifest['artifacts'].items():assert sha((root/'installed'/name).read_bytes())==expected
    assert ids['missing_assembler_input']=={'path':'/tmp/ccA4aFXT.s','exists_now':False}
print('I verified original seed/Stage1/translation passes followed by a missing assembler-input failure, unchanged sources and matching retained tool/module identities. No fixed-point/native-suite/release acceptance follows.')
