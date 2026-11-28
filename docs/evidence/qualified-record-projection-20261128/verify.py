"""I verify my original native cohort archives without executing a product."""
import subprocess,sys
import argparse, gzip, hashlib, json, re, tempfile, zipfile
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--repo',type=Path,required=True)
args=parser.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
raw=(args.archive/'manifest.json').read_bytes()
assert sha(raw)=='4e97519e53c8087846b43fac657f253af9eeb7399bb2c7188296c54f34463214'
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
    for label in ['linux','darwin']:
        expected=json.loads((root/('verified-'+label+'.json')).read_text())
        result=subprocess.run([sys.executable,str(root/'verify_projection.py'),'--root',str(root),'--repo',str(args.repo),'--platform',label],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert json.loads(result.stdout)==expected
        assert expected['focused_projection_gate_passed'] and expected['commands']==48 and not expected['whole_join_runtime_gate'] and not expected['whole_release']
        print(result.stdout,end='')
    result=subprocess.run([sys.executable,str(root/'verify_refusals.py'),'--root',str(root)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    bridge=json.loads((root/'component-source-bridge.json').read_text())
    paths=subprocess.check_output(['git','diff','--name-only',bridge['focused_source'],bridge['full_gate_source']],cwd=args.repo,text=True).splitlines()
    assert paths==bridge['changed_paths']==['Makefile.gnu','docs/ROADMAP.md','tests/nvm2c_representation_joins_reference.c','tests/test_nvm2c_representation_join_allocation.py']
    assert not bridge['qualification_transferred'] and bridge['fresh_full_gate_required'] and not bridge['whole_release']
    print(result.stdout,end='')
print('I verify paired focused C projection repair, strict old-code controls and preserved observer refusals. Full join runtime, exact release candidate and all69 release acceptance remain open.')
