from pathlib import Path
import hashlib,tarfile,shutil,json,inspect
assert 'filter' in inspect.signature(tarfile.TarFile.extractall).parameters
archive=Path('/tmp/nanolang-file-native-4e8-source.tar.gz');assert hashlib.sha256(archive.read_bytes()).hexdigest()=='e67142efe9259d180295a07d40cd0b6b643eccc5d3596eb605949033ee543cbe'
root=Path('/tmp/nanolang-file-native-4e8-corrected');root.mkdir(exist_ok=False)
with tarfile.open(archive) as t:t.extractall(root,filter='data')
source=json.loads(Path('/tmp/nanolang-file-native-4e8-source-map.json').read_text())
for name,sha in source.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
shutil.copyfile('/tmp/nanolang-file-native-4e8-tracked',root/'.qualification-tracked')
(root/'.qualification-pin').write_text('4e8ec0d67\n')
assert not (root/'obj').exists();print('Verified source archive and',len(source),'source hashes; fresh provider closure')
