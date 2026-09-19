import hashlib,json,pathlib,shutil,sys,tarfile
platform=sys.argv[1];base=pathlib.Path(sys.argv[2]);bootstrap=pathlib.Path(sys.argv[3]);gate=pathlib.Path(sys.argv[4]);out=pathlib.Path(sys.argv[5]);out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert read(gate/'status.json')['success'] and read(gate/'status.json')['testsRun']==6
assert all(v['status']==0 for v in read(bootstrap/'status.json')['steps'])
for d in (bootstrap,gate):
 for a,b in [('source-before.json','source-after.json'),('host-tools-before.json','host-tools-after.json')]:assert read(d/a)==read(d/b)
 for p,v in read(d/'source-after.json').items():assert sha(base/p)==v,p
 for v in read(d/'host-tools-after.json').values():assert sha(pathlib.Path(v['path']))==v['sha256']
assert read(gate/'inputs-after-setup.json')==read(gate/'inputs-after-tests.json')
for p,v in read(gate/'inputs-after-tests.json').items():assert sha(base/p)==v,p
for p,v in read(gate/'producers-after-setup.json').items():assert sha(pathlib.Path(p))==v,p
native=read(gate/'native-compiler.json');assert sha(pathlib.Path(native['path']))==native['sha256']
artifacts={}
for name,d in [('bootstrap',bootstrap),('gates',gate)]:
 dest=out/name;dest.mkdir()
 for p in d.iterdir():
  if p.is_file():shutil.copy2(p,dest/p.name)
 for p in d.rglob('*'):
  if p.is_file() and p.parent!=d:artifacts[str(p)]=sha(p)
log=pathlib.Path(str(gate)+'.log');shutil.copy2(log,out/'gates/runner.log')
archive=out.parent/('nanolang-owner-array-source-final-'+platform+'-artifacts.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 t.add(bootstrap,arcname='bootstrap');t.add(gate,arcname='gates');t.add(log,arcname='gates/runner.log')
write(out/'artifact-sha256.json',artifacts)
write(out/'summary.json',{'platform':platform,'pin':'1c4e29067aa21ee843e00997d42851877f7082fe','source_tree':str(base),'bootstrap':read(bootstrap/'status.json'),'gates':read(gate/'status.json'),'sources':len(read(gate/'source-after.json')),'post_setup_inputs':len(read(gate/'inputs-after-tests.json')),'producers':len(read(gate/'producers-after-setup.json')),'tools':len(read(gate/'host-tools-after.json')),'native_compiler':native,'archive':{'path':str(archive),'sha256':sha(archive)},'artifact_files':len(artifacts)})
shutil.copy2(__file__,out/pathlib.Path(__file__).name)
reports={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};write(out/'report-sha256.json',reports)
print(json.dumps({'platform':platform,'reports':len(reports),'artifacts':len(artifacts),'archive':str(archive),'sha256':sha(archive)}))
