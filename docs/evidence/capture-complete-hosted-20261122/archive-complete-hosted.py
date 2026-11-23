from pathlib import Path
import gzip,hashlib,json
ROOT=Path.cwd();OUT=ROOT/'.test_output/capture-repair-20261122';target=ROOT/'docs/evidence/capture-complete-hosted-20261122';assert not (target/'manifest.json').exists();target.mkdir(parents=True,exist_ok=True);(target/'inputs').mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
shared={}
for manifest in (ROOT/'docs/evidence').glob('*/manifest.json'):
 if manifest.parent==target:continue
 data=json.loads(manifest.read_text())
 for e in data.get('files',[]):
  if all(k in e for k in ('sha256','artifact')):shared[e['sha256']]=(manifest.parent/e['artifact']).resolve()
rows=[];links={}
for directory in ('hosted-fixed-45009129801','hosted-clang-45009127688','userguide-45010076831-default-refusal'):
 for p in sorted((OUT/directory).rglob('*')):
  if p.is_symlink():links[str(p.relative_to(OUT))]=str(p.readlink())
  if p.is_file():rows.append(('observer/'+str(p.relative_to(OUT)),p.read_bytes()))
for name in ('verify-peer-eof-archive.py', 'peer-eof-archive-verification.json', 'audit-hosted-fixed.py', 'audit-hosted-clang.py', 'hosted-fixed-45009129801-independent-audit.json', 'hosted-clang-45009127688-independent-audit.json', 'userguide-concurrency-live.json', 'userguide-45010076831-source-audit.json', 'userguide-45010076831-failed.log', 'userguide-45009818891-build-cancelled.log', 'userguide-45009818891-log-refusal.txt', 'clang-complete-45009127688.log', 'fixed-producer-45009129801.log', 'fixed-complete-45009129801.log'):
 rows.append(('observer/'+name,(OUT/name).read_bytes()))
for name in ('src/module_builder.c','tests/test_module_generation_probe.c','tests/test_source_snapshots.py','tests/test_response_driver_grammar.py','scripts/ci_qualify_as246.py','.github/workflows/source-capture-toolchain-diagnostic.yml','Makefile.gnu'):
 rows.append(('corrected/'+name,(ROOT/name).read_bytes()))
entries=[]
for label,raw in rows:
 h=sha(raw)
 if h in shared:
  p=shared[h];artifact='../'+str(p.relative_to(ROOT/'docs/evidence'));blob=p.read_bytes();assert gzip.decompress(blob)==raw
 else:
  artifact='inputs/'+h+'.gz';p=target/artifact
  if not p.exists():p.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
  blob=p.read_bytes()
 entries.append(dict(path=label,sha256=h,bytes=len(raw),artifact=artifact,compressed_sha256=sha(blob),compressed_bytes=len(blob)))
(target/'manifest.json').write_text(json.dumps(dict(schema='nanolang-retained-evidence/v1',scope='I retain complete private Clang21/GNU2.42 and fixed-GCC/r1 cohorts at3ef, native peer verification and the separate ordinary system-tool PR refusal. Fixedr1 with external Clang, production adoption and current-head paired release acceptance remain separate.',symlink_targets=links,files=entries),indent=2)+'\n')
for e in entries:
 blob=(target/e['artifact']).read_bytes();raw=gzip.decompress(blob);assert sha(blob)==e['compressed_sha256'] and len(blob)==e['compressed_bytes'] and sha(raw)==e['sha256'] and len(raw)==e['bytes']
a=dict(logical_files=len(entries),raw_bytes=sum(e['bytes'] for e in entries),local_payloads=len(list((target/'inputs').glob('*.gz'))),local_compressed_bytes=sum(p.stat().st_size for p in (target/'inputs').glob('*.gz')),all_raw_and_compressed_hashes_and_lengths_match=True)
(target/'archive-audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
