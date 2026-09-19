import hashlib,json,pathlib,shutil,tarfile,subprocess
root=pathlib.Path('/home/jkh/Src/nanolang-mutation-builtin-corrected')
out=root/'docs/evidence/mutation-builtin-identity';out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
runs={
 'first-bootstrap':pathlib.Path('/tmp/nanolang-mutation-identity-66a-bootstrap'),
 'bootstrap':pathlib.Path('/tmp/nanolang-mutation-identity-82ce-bootstrap'),
 'first-focused':pathlib.Path('/tmp/nanolang-mutation-identity-54ea-focused'),
 'corrected':pathlib.Path('/tmp/nanolang-mutation-identity-95a1-corrected'),
 'nested':pathlib.Path('/tmp/nanolang-mutation-identity-0e4d-nested')}
for name,path in runs.items():
 assert path.is_dir()
 target=out/name;target.mkdir()
 for file in path.iterdir():
  if file.is_file():shutil.copy2(file,target/file.name)
fixtures=[pathlib.Path(x) for x in ['/tmp/nanolang-mutation-identity-7i1p1jr6','/tmp/nanolang-mutation-unbound-corrected-x7nbdh_l','/tmp/nanolang-mutation-nested-wj9cd10u','/tmp/nano-reduce-source-v89c5r13','/tmp/nano-reduce-source-co1fhyqg']]
artifacts={str(p):sha(p) for d in fixtures for p in d.rglob('*') if p.is_file()}
assert len(artifacts)>40
save(out/'artifacts.json',artifacts)
archive=pathlib.Path('/tmp/nanolang-mutation-identity-qualification.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for path in list(runs.values())+fixtures:tar.add(path,arcname=path.name)
save(out/'archive.json',{'path':str(archive),'sha256':sha(archive),'size':archive.stat().st_size})
final=json.loads((runs['nested']/'sources-after.json').read_text())
assert all(sha(root/p)==h for p,h in final.items())
bootstrap=json.loads((runs['bootstrap']/'built-tools.json').read_text())
assert all(sha(root/p)==h for p,h in bootstrap.items())
for label in ('bootstrap','first-focused','corrected','nested'):
 path=runs[label]
 assert json.loads((path/'sources-before.json').read_text())==json.loads((path/'sources-after.json').read_text())
 assert json.loads((path/'tools-before.json').read_text())==json.loads((path/'tools-after.json').read_text())
assert json.loads((runs['nested']/'checker-drivers-before.json').read_text())==json.loads((runs['nested']/'checker-drivers-after.json').read_text())
current={str(p):sha(p) for p in (root/'bin').iterdir() if p.is_file()}
save(out/'final-tools.json',current)
shutil.copy2('/tmp/owned-array-mutation-binding-focused-task.txt',out/'pending-ledger-description.txt')
shutil.copy2(__file__,out/'seal-runner.py')
reports={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
save(out/'manifest.json',{'pin':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'reports':reports,'report_count':len(reports),'artifact_count':len(artifacts),'source_count':len(final),'final_tool_count':len(current),'compiler_production_unchanged_since_bootstrap':not subprocess.check_output(['git','diff','82ce8b29d..HEAD','--','src','src_nano','runtime','modules'],cwd=root),'limits':['The first focused suite was not globally passing; only three methods passed before the unbound fixture parse failure.','I reran only the corrected unbound method and previously unrun adjacency, then added six nested checker cases.','TemporaryDirectory conversion artifacts were removed by the unchanged existing harness after successful commands.','Object/cache inputs may grow during checker-driver setup; I preserve separate snapshots without asserting whole-map equality.','Linux-only checker qualification; no owner ARRAY source/public runtime activation.']})
print(json.dumps({'reports':len(reports),'artifacts':len(artifacts),'sources':len(final),'tools':len(current),'archive_sha256':sha(archive)},indent=2))
