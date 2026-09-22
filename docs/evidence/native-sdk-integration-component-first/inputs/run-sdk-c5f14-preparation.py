import pathlib,sys,os,json,hashlib,shutil,subprocess
root=pathlib.Path(sys.argv[1]).resolve();report=pathlib.Path(sys.argv[2]);host=sys.argv[3];manifest=pathlib.Path(sys.argv[4]);report.mkdir();sys.path.insert(0,str(root))
from tests.native_sdk_runner import run
selected=json.loads(manifest.read_text())
def dump(name,value):(report/name).write_text(json.dumps(value,indent=2)+'\n')
def identity(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(1048576),b''):h.update(data)
 return dict(sha256=h.hexdigest(),bytes=p.stat().st_size,mode=p.stat().st_mode&0o7777)
def sources():
 result={}
 for row in selected['records']:
  actual=identity(root/row['path']);assert actual==dict(sha256=row['sha256'],bytes=row['bytes'],mode=int(row['mode'],8)&0o7777),row['path'];result[row['path']]=actual
 return result
def products():return {str(p.relative_to(root)):identity(p) for name in ('bin','obj','lib') for p in sorted((root/name).rglob('*')) if p.is_file()}
cc='/usr/bin/gcc' if host=='linux' else '/usr/bin/clang'
env=dict(CC=cc,NANO_CC=cc,NANOLANG_SDK_ROOT=str(root),NANO_SHADOW_TIMING=None,NANO_SHADOW_TIMEOUT_SECONDS=None,NANO_SDK_MIN_FREE_BYTES=str(2*1024**3))
def command(name,args,timeout=180):
 print(name,flush=True);return run(report,name,args,root,env,timeout=timeout,track_descendants=True)
if host=='puck':
 env.update(PATH='/opt/homebrew/bin:'+os.environ['PATH'],PKG_CONFIG_PATH='/opt/homebrew/opt/libffi/lib/pkgconfig:/opt/homebrew/opt/openssl@3/lib/pkgconfig',LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib',DYLD_LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib')
 out,_,_=command('sdk-path',['/usr/bin/xcrun','--show-sdk-path']);env['SDKROOT']=out.decode().strip()
toolpaths=[pathlib.Path(cc).resolve(),pathlib.Path(sys.executable).resolve()]
for name in ('make','ar','ld','nm','pkg-config'):
 path=shutil.which(name,path=env.get('PATH',os.environ['PATH']));assert path,name;toolpaths.append(pathlib.Path(path).resolve())
if host=='puck':out,_,_=command('compiler-owner',['/usr/bin/xcrun','--find','clang'])
else:out,_,_=command('compiler-owner',[cc,'-print-prog-name=cc1'])
toolpaths.append(pathlib.Path(out.decode().strip()).resolve())
def tools():return {str(p):identity(p) for p in sorted(set(toolpaths))}
command('compiler-version',[cc,'--version'])
before=sources();tb=tools();dump('source-before.json',before);dump('tools-before.json',tb);dump('products-before.json',products());dump('configuration.json',dict(pin=selected['pin'],host=host,environment=env,scope='fresh integrated build and C-seed/S1/S2 bootstrap; original compiler shadow bounds; ordinary providers'))
first=None
try:
 command('build',['make','-j2','CC='+cc,'build'],3600)
 command('bootstrap',['make','-j2','CC='+cc,'bootstrap'],3600)
except BaseException as error:
 first=repr(error);dump('first-failure.json',dict(error=first));raise
finally:
 after=sources();ta=tools();dump('source-after.json',after);dump('tools-after.json',ta);dump('products-after.json',products());dump('terminal.json',dict(first_failure=first,source_equal=before==after,tools_equal=tb==ta));assert before==after and tb==ta
print('PASS',flush=True)
