from pathlib import Path
import gzip,hashlib,json
ROOT=Path.cwd();OUT=ROOT/'.test_output/capture-repair-20261122';target=ROOT/'docs/evidence/capture-fixed-clang-timeout-20261122';assert not (target/'manifest.json').exists();target.mkdir(parents=True,exist_ok=True);(target/'inputs').mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
shared={}
for manifest in (ROOT/'docs/evidence').glob('*/manifest.json'):
 if manifest.parent==target:continue
 data=json.loads(manifest.read_text())
 for e in data.get('files',[]):
  if all(k in e for k in ('sha256','artifact')):shared[e['sha256']]=(manifest.parent/e['artifact']).resolve()
rows=[];links={}
for directory in ('hosted-fixed-clang-45013684614','fixed-clang-45013684614-partial'):
 for p in sorted((OUT/directory).rglob('*')):
  if p.is_symlink():links[str(p.relative_to(OUT))]=str(p.readlink())
  if p.is_file():rows.append(('observer/'+str(p.relative_to(OUT)),p.read_bytes()))
for name in ('fixed-clang-45013684614-failed.log', 'fixed-clang-45013684614-partial-audit.json', 'read-fixed-clang-partial.py', 'provisioner-all-controls.log', 'provisioner-reuse-controls.log'):
 rows.append(('observer/'+name,(OUT/name).read_bytes()))
for p in sorted((OUT/'profile-observer-darwin-control').rglob('*')):
 if p.is_file() and ('root' not in p.relative_to(OUT/'profile-observer-darwin-control').parts or p.name=='test_module_generation_probe' or p.name in ('__init__.py','test_source_snapshots.py','characterize_source_snapshot.py','characterize_linker_inputs.py','test_module_cache_publication.py','test_bytecode_shadows.py','native_process.py')):
  rows.append(('observer/'+str(p.relative_to(OUT)),p.read_bytes()))
import subprocess
for name in ('src/module_builder.c','tests/test_source_snapshots.py','tests/test_module_generation_probe.c','scripts/ci_qualify_as246.py','scripts/ci_fixed_clang_inputs.py','.github/workflows/source-capture-toolchain-diagnostic.yml','tests/test_fixed_clang_inputs.py','Makefile.gnu'):
 rows.append(('executed-source/a67ffa59283ff398f3afbc1d3d746e742d489bcd/'+name,subprocess.check_output(['git','show','a67ffa59283ff398f3afbc1d3d746e742d489bcd:'+name])))
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
(target/'manifest.json').write_text(json.dumps(dict(schema='nanolang-retained-evidence/v1',scope='I retain failed combined Clang/r1 source timeout45013684614 with its exact inputs/native images, prior completed producer and actual partial stream. The full source deadline, later cohorts and production admission remain unmet. I retain only a separate actual Darwin observer control as harness validation.',symlink_targets=links,files=entries),indent=2)+'\n')
for e in entries:
 blob=(target/e['artifact']).read_bytes();raw=gzip.decompress(blob);assert sha(blob)==e['compressed_sha256'] and len(blob)==e['compressed_bytes'] and sha(raw)==e['sha256'] and len(raw)==e['bytes']
a=dict(logical_files=len(entries),raw_bytes=sum(e['bytes'] for e in entries),local_payloads=len(list((target/'inputs').glob('*.gz'))),local_compressed_bytes=sum(p.stat().st_size for p in (target/'inputs').glob('*.gz')),all_raw_and_compressed_hashes_and_lengths_match=True)
(target/'archive-audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
