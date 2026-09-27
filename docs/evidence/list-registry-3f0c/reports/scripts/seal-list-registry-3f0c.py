from pathlib import Path
import hashlib,json,gzip,os
root=Path('/home/jkh/Src/nanolang-generic-list-mutations')
dest=root/'docs/evidence/list-registry-3f0c';dest.mkdir(parents=True,exist_ok=False)
cas=Path('/tmp/nanolang-list-registry-3f0c-artifacts');cas.mkdir(exist_ok=False)
source=Path('/tmp/nanolang-list-registry-3f0c-linux')
manifest=dict(source_pin='3f0ca4d788ad10722139f77f2eb89cd722d591b8',production_pin='40870d07805c85c2a6600c7d02b8440cd547f57a',scope='One private full typecheck_driver diagnostic, original10s shadow deadline and120s outer bound. No production acceptance. Final unmatched interval is unmeasured; recursive view timing is inclusive and nonadditive.',reports={},artifacts={},commands=[])
def report(p,label):
 b=p.read_bytes();raw=hashlib.sha256(b).hexdigest();n=len(b)
 rel=Path('reports')/label/p.name
 if len(b)>65536:rel=Path(str(rel)+'.gz');b=gzip.compress(b,mtime=0)
 q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 manifest['reports'][str(rel)]=dict(sha256=raw,bytes=n,stored_sha256=hashlib.sha256(b).hexdigest(),stored_bytes=len(b),original_path=str(p))
for p in sorted(source.iterdir()):
 if p.is_file() and p.suffix in ('.json','.jsonl','.log','.mk'):report(p,'diagnostic')
for p in sorted((source/'objects').iterdir()):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==p.name
 os.link(p,cas/p.name);manifest['artifacts'][p.name]=dict(bytes=len(b),path=str(cas/p.name))
for p in sorted(source.glob('*-status.json')):
 s=json.loads(p.read_text());before=json.loads((source/p.name.replace('-status','-before')).read_text());after=json.loads((source/p.name.replace('-status','-after')).read_text());assert before==after and s['inputs_equal'] and s['reaped'] and s['group_gone'] and not s['timeout']
 manifest['commands'].append({k:s[k] for k in ('returncode','seconds','inputs_equal','reaped','group_gone','timeout')}|{'name':p.name})
assert json.loads((source/'inputs-before-run.json').read_text())==json.loads((source/'inputs-after.json').read_text())
for p in ('/tmp/run-list-registry-3f0c.py','/tmp/summarize-typecheck-timing.py',__file__):report(Path(p),'scripts')
manifest.update(report_count=len(manifest['reports']),unique_artifacts=len(manifest['artifacts']),artifact_bytes=sum(x['bytes'] for x in manifest['artifacts'].values()),command_pairs=len(manifest['commands']))
(dest/'seal.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({k:manifest[k] for k in ('report_count','unique_artifacts','artifact_bytes','command_pairs')})
