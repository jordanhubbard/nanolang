from pathlib import Path
import hashlib,json,os,shutil,sys,tarfile
old=Path('/tmp/nanolang-native-sdk-fe645-linux-source');root=Path('/tmp/nanolang-native-sdk-fe645-linux-diagnostic-source');report=Path('/tmp/nanolang-native-sdk-fe645-linux-diagnostic');prep=Path('/tmp/nanolang-native-sdk-fe645-linux-preparation');root.mkdir();report.mkdir()
def ident(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return dict(sha256=h.hexdigest(),bytes=p.stat().st_size,mode=p.stat().st_mode&0o7777)
def dump(n,v):(report/(n+'.json')).write_text(json.dumps(v,indent=2)+'\n')
selected=json.loads(Path('/tmp/native-sdk-fe645-selected-git.json').read_text());oldsources=json.loads((prep/'source-after.json').read_text());oldproducts=json.loads((prep/'products-after.json').read_text());oldtools=json.loads((prep/'tools-after.json').read_text())
assert all(ident(old/p)==v for p,v in oldsources.items());assert all(ident(old/p)==v for p,v in oldproducts.items());assert all(ident(Path(p))==v for p,v in oldtools.items())
assert shutil.disk_usage(root).free>2*1024**3+sum(v['bytes'] for v in oldproducts.values())
with tarfile.open('/tmp/native-sdk-fe645-source.tar.gz') as a:a.extractall(root,filter='data')
for name in ('bin','obj','lib'):
 if (old/name).exists():shutil.copytree(old/name,root/name,symlinks=True,dirs_exist_ok=True)
def sources():return {p:ident(root/p) for p in oldsources}
def products():return {str(p.relative_to(root)):ident(p) for name in ('bin','obj','lib') for p in sorted((root/name).rglob('*')) if p.is_file()}
def tools():return {p:ident(Path(p)) for p in oldtools}
assert sources()==oldsources;assert products()==oldproducts;assert tools()==oldtools
for row in selected['records']:assert ident(root/row['path'])==dict(sha256=row['sha256'],bytes=row['bytes'],mode=int(row['mode'],8)&0o7777)
dump('source-before',sources());dump('products-before',products());dump('tools-before',tools());dump('copy-proof',dict(pin=selected['pin'],original_source=str(old),original_report=str(prep),symlinks={str(p.relative_to(root)):os.readlink(p) for p in root.rglob('*') if p.is_symlink()},scope='one diagnostic on byte-identical copied Cseed/full native providers and source; no compiler refresh'))
sys.path.insert(0,str(root));from tests.native_sdk_runner import run
private=report/'tmp';private.mkdir();output=report/'stage1';sentinel=b'diagnostic output sentinel\n';output.write_bytes(sentinel)
env=dict(CC='/usr/bin/gcc',NANO_CC='/usr/bin/gcc',NANOLANG_SDK_ROOT=str(root),TMPDIR=str(private),NANO_SHADOW_TIMING='1',NANO_SHADOW_TIMEOUT_SECONDS=None,NANO_SDK_MIN_FREE_BYTES=str(2*1024**3))
first=None
try:
 _,_,status=run(report,'timed-bootstrap',[root/'bin/nanoc_c','src_nano/nanoc_v06.nano','-o',output,'--verbose','--llm-shadow-json',report/'shadows.json'],root,env,expected=(0,1),timeout=120,track_descendants=True)
 if status['returncode']!=0:assert output.read_bytes()==sentinel
 dump('output',dict(identity=ident(output),sentinel_preserved=output.read_bytes()==sentinel))
except BaseException as error:first=repr(error);dump('first-failure',dict(error=first));raise
finally:
 sa=sources();ta=tools();pa=products();dump('source-after',sa);dump('tools-after',ta);dump('products-after',pa);dump('terminal',dict(first_failure=first,source_equal=sa==oldsources,tools_equal=ta==oldtools,original_products_unchanged=all(ident(old/p)==v for p,v in oldproducts.items()),retained_product_members_equal=all(pa.get(p)==v for p,v in oldproducts.items())))
 assert sa==oldsources and ta==oldtools and all(ident(old/p)==v for p,v in oldproducts.items())
