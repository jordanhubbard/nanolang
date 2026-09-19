import hashlib,json,pathlib,shutil,subprocess,tarfile
root=pathlib.Path('/home/jkh/Src/nanolang-owned-array-source-refusal-phase');dest=root/'docs/evidence/owned-array-source-linux';dest.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
runs={'bootstrap':pathlib.Path('/tmp/nanolang-owner-array-source-c9fb-bootstrap'),'first':pathlib.Path('/tmp/nanolang-owner-array-source-c9fb-gates'),'phase':pathlib.Path('/tmp/nanolang-owner-array-source-b9ecc-remaining'),'remaining':pathlib.Path('/tmp/nanolang-owner-array-source-32f2-remaining')}
artifacts={}
for name,path in runs.items():
 target=dest/name;target.mkdir()
 for p in path.iterdir():
  if p.is_file():shutil.copy2(p,target/p.name)
 for p in path.rglob('*'):
  if p.is_file() and p.parent!=path:artifacts[str(p)]=sha(p)
 log=pathlib.Path(str(path)+'.log')
 if log.exists():shutil.copy2(log,target/'runner.log')
for name,path in runs.items():
 for first,last in [('source-before.json','source-after.json'),('host-tools-before.json','host-tools-after.json'),('tools-before.json','tools-after.json'),('producers-before.json','producers-after.json')]:
  if (path/first).exists():assert json.loads((path/first).read_text())==json.loads((path/last).read_text())
 if name in ('phase','remaining'):assert json.loads((path/'inputs-before.json').read_text())==json.loads((path/'inputs-after.json').read_text())
assert json.loads((runs['first']/'inputs-after-setup.json').read_text())==json.loads((runs['first']/'inputs-after-tests.json').read_text())
archive=pathlib.Path('/tmp/nanolang-owner-array-source-linux-artifacts.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for name,path in runs.items():tar.add(path,arcname=name)
 for name in ('first','phase','remaining'):tar.add(pathlib.Path(str(runs[name])+'.log'),arcname=name+'/runner.log')
summary={'source_pin':'c9fb07ed55329f59e6b67ef0bf3f9dc756d6aa88','fixture_pin':'32f2f1398','bootstrap_pass_seconds':271.5786423999816,'tools_setup_pass_seconds':26.522672194987535,'producer_setup_seconds':177.67095257801702,'first_terminal':{'testsRun':5,'passed_methods':4,'failures':1,'seconds':365.579,'failure':'optional-local raw emitter phase expectation'},'second_terminal':{'testsRun':1,'failures':1,'seconds':0.233,'passed_groups':['optional-local','optional-add','optional-negate','bound-at'],'failure':'reserved not parser phase expectation'},'remaining_pass':json.loads((runs['remaining']/'status.json').read_text()),'archive':{'path':str(archive),'sha256':sha(archive)},'sources':2265,'hosttools':7,'post_setup_inputs':799,'retained_producer_tools':13,'artifact_files':len(artifacts),'limits':['Linux GCC only','No Darwin claim','No final canonical parser/File integration claim','Mutation and full parents remain open']}
(dest/'artifact-sha256.json').write_text(json.dumps(artifacts,indent=2)+'\n');(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');shutil.copy2(__file__,dest/pathlib.Path(__file__).name)
reports={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()};(dest/'report-sha256.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps({'reports':len(reports),'artifacts':len(artifacts),'archive':summary['archive']}))
