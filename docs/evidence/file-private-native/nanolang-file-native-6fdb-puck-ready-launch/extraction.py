from pathlib import Path
import hashlib,tarfile,shutil,json
archive=Path('/tmp/nanolang-file-native-6fdb-source.tar.gz');assert hashlib.sha256(archive.read_bytes()).hexdigest()=='4a182acdc2dca2c4806685c7abaab2adeed1b4f8ca735f4b888d83c64e482bf5'
root=Path('/tmp/nanolang-file-native-6fdb-ready').resolve();root.mkdir(exist_ok=False)
with tarfile.open(archive) as t:t.extractall(root,filter='data')
source=json.loads(Path('/tmp/nanolang-file-native-6fdb-source-map.json').read_text())
for name,sha in source.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
shutil.copyfile('/tmp/nanolang-file-native-6fdb-tracked',root/'.qualification-tracked');(root/'.qualification-pin').write_text('6fdb7a60e\n')
prior=Path('/tmp/nanolang-file-native-4e8-corrected').resolve();inputs=json.loads(Path('/tmp/nanolang-file-native-4e8-puck-corrected/normal-inputs-before.json').read_text());copied={}
for name,d in inputs.items():
 p=Path(name).resolve();assert hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256'],name
 relative=p.relative_to(prior);q=root/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert hashlib.sha256(q.read_bytes()).hexdigest()==d['sha256'];copied[str(relative)]=d['sha256']
Path('/tmp/nanolang-file-native-6fdb-puck-ready-provider-reuse.json').write_text(json.dumps({'setup_pin':'4e8ec0d67','corrected_pin':'6fdb7a60e','objects':copied},indent=2))
print('Verified sources',len(source),'and reused providers',len(copied))
