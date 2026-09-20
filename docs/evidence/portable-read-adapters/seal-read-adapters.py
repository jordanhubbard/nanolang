import pathlib,json,hashlib,shutil,tarfile
repo=pathlib.Path('/home/jkh/Src/nanolang-portable-read-adapters');dest=repo/'docs/evidence/portable-read-adapters';dest.mkdir(exist_ok=False);store=pathlib.Path('/tmp/nanolang-read-adapter-artifacts');store.mkdir(exist_ok=False)
remote=pathlib.Path('/tmp/nanolang-read-adapter-puck-downloaded');remote.mkdir(exist_ok=False)
with tarfile.open('/tmp/nanolang-read-adapter-puck-seal.tar.gz') as t:t.extractall(remote,filter='data')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
art={};summary={}
for host,root in [('linux',pathlib.Path('/tmp/nanolang-read-adapter-linux-seal')),('puck',remote/'puck')]:
 out=dest/host;out.mkdir();shutil.copytree(root/'reports',out/'reports')
 for name in ['summary.json','current.json','artifact-store.json','report-sha256.json','collector.py']:shutil.copyfile(root/name,out/name)
 for path in (root/'objects').iterdir():
  digest=sha(path);assert digest==path.name
  if digest not in art:shutil.copyfile(path,store/digest);art[digest]={'path':str(store/digest),'bytes':path.stat().st_size}
 summary[host]=json.loads((root/'summary.json').read_text())
for p in ['/tmp/launch-read-adapter-continuation.py','/tmp/run-read-adapter-gates.py','/tmp/collect-read-adapter-seal.py','/tmp/seal-read-adapters.py']:shutil.copyfile(p,dest/pathlib.Path(p).name)
archive=pathlib.Path('/tmp/nanolang-read-adapter-puck-seal.tar.gz')
summary.update(unique_artifacts=len(art),artifact_bytes=sum(v['bytes'] for v in art.values()),puck_archive={'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
(dest/'qualification-summary.json').write_text(json.dumps(summary,indent=2)+'\n');(dest/'artifact-store.json').write_text(json.dumps(art,indent=2)+'\n')
manifest={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()};(dest/'report-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'reports':len(manifest),'artifacts':len(art),'bytes':summary['artifact_bytes'],'linux_pairs':len(summary['linux']['pairs']),'puck_pairs':len(summary['puck']['pairs'])}))
