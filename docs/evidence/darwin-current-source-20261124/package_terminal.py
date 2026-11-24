"""I archive only my terminal host-local Darwin qualification into a new root."""
from pathlib import Path
import os,json,hashlib,gzip,tempfile,time,subprocess,stat
root=Path('/private/tmp/nanolang-source-qual-20261124-tgs3q7n7')
q=json.loads((root/'result.json').read_text());cfg=json.loads((root/'config.json').read_text())
assert q['terminal'], 'I do not archive a live qualification.'
pid=json.loads((root/'launch.json').read_text())['controller_pid']
process=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid=,state=,command='],capture_output=True,text=True,timeout=10)
assert process.returncode!=0 or ' Z' in process.stdout,process.stdout
out=Path(tempfile.mkdtemp(prefix='nanolang-darwin-evidence-',dir='/private/tmp'));objects=out/'objects';objects.mkdir();sha=lambda b:hashlib.sha256(b).hexdigest();manifest=dict(format=1,source_pin=cfg['pin'],qualification_root=str(root),archive_root=str(out),controller_exit=q['controller_exit'],files=[],symlinks=[],chunks={},source_guards={},tool_guards={},archiver_seconds=None)
def record(path,name):
 digest=hashlib.sha256();size=0;chunks=[]
 with path.open('rb') as f:
  while True:
   raw=f.read(8*1024*1024)
   if not raw:break
   digest.update(raw);size+=len(raw);h=sha(raw);chunks.append(h)
   if h not in manifest['chunks']:
    zipped=gzip.compress(raw,compresslevel=6,mtime=0);target=objects/(h+'.gz');target.write_bytes(zipped);manifest['chunks'][h]=dict(bytes=len(raw),compressed_bytes=len(zipped),compressed_sha256=sha(zipped))
 manifest['files'].append(dict(path=name,bytes=size,sha256=digest.hexdigest(),mode=stat.S_IMODE(path.stat().st_mode),chunks=chunks))
start=time.monotonic()
for directory,dirs,names in os.walk(root,followlinks=False):
 dirs[:]=sorted(n for n in dirs if n!='.git')
 for name in dirs:
  path=Path(directory)/name
  if path.is_symlink():manifest['symlinks'].append(dict(path=str(path.relative_to(root)),target=os.readlink(path)))
 for name in sorted(names):
  path=Path(directory)/name;rel=path.relative_to(root)
  if name=='.git':continue
  if path.is_symlink():manifest['symlinks'].append(dict(path=str(rel),target=os.readlink(path)));continue
  if path.is_file():record(path,'qualification/'+str(rel))
for name in cfg['sources']:
 path=root/'source-normal'/name;actual=sha(path.read_bytes());manifest['source_guards'][name]=dict(expected=cfg['sources'][name],actual=actual)
for name,expected in q['tools_before'].items():
 path=Path(name);actual=sha(path.read_bytes());manifest['tool_guards'][name]=dict(expected=expected,actual=actual,resolved=str(path.resolve()));record(path,'tool/'+name.lstrip('/'))
manifest['archiver_seconds']=time.monotonic()-start
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(archive=str(out),records=len(manifest['files']),chunks=len(manifest['chunks']),seconds=manifest['archiver_seconds'],manifest_sha256=sha((out/'manifest.json').read_bytes()))),flush=True)
