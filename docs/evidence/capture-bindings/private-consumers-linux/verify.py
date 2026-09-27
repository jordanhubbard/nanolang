import hashlib,json,sys,tarfile
from pathlib import Path
p=Path(sys.argv[1]);archive=p/'capture-private-b995-evidence.tar.gz';manifest=json.loads((p/'capture-private-b995-manifest.json').read_text());expected='5a4d541b0273dbae69e72ef124e7c1b992796fbe8e190bef47a243c44cf32bf1'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers();assert len(members)==len(manifest) and {m.name for m in members}==set(manifest)
 for m in members:
  entry=manifest[m.name];assert m.isfile() and m.size==entry['size'] and m.mode==entry['mode']
  assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==entry['sha256']
print(json.dumps({'status':'PASS','archive_sha256':expected,'members':len(manifest)}))
