import pathlib,sys,os,json,hashlib,shutil,subprocess,shlex
root=pathlib.Path(sys.argv[1]).resolve();report=pathlib.Path(sys.argv[2]);host=sys.argv[3];manifest=pathlib.Path(sys.argv[4]);prep=pathlib.Path(sys.argv[5]);oldroot=pathlib.Path(sys.argv[6]).resolve();report.mkdir();sys.path.insert(0,str(root))
from tests.native_sdk_runner import run
selected=json.loads(manifest.read_text())
private_tmp=report/'tmp';private_tmp.mkdir()
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
env=dict(TMPDIR=str(private_tmp),CC=cc,NANO_CC=cc,NANOLANG_SDK_ROOT=str(root),NANO_SHADOW_TIMING=None,NANO_SHADOW_TIMEOUT_SECONDS=None,NANO_SDK_MIN_FREE_BYTES=str(2*1024**3))
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
before=sources();tb=tools();dump('source-before.json',before);dump('tools-before.json',tb);dump('products-before.json',products());dump('configuration.json',dict(pin=selected['pin'],host=host,environment=env,scope='current Cseed primitive and actual split ownership controls with verified copied unchanged4e compiler/providers; not full bootstrap'))
first=None
try:
 oldsources=json.loads((prep/'source-after.json').read_text())
 assert all(identity(oldroot/p)==value for p,value in oldsources.items())
 changed=[p for p,value in oldsources.items() if p in before and before[p]!=value]
 assert [p for p in changed if p.startswith(('src/','src_nano/','modules/','schema/','scripts/'))]==[],changed
 assert tb==json.loads((prep/'tools-after.json').read_text())
 oldproducts=json.loads((prep/'products-after.json').read_text())
 assert all(identity(oldroot/p)==value for p,value in oldproducts.items())
 for p,value in oldproducts.items():
  target=root/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(oldroot/p,target);assert identity(target)==value,p
 dump('copied-products-before.json',products())
 dump('provider-reuse-proof.json',dict(old_pin='4e4a17be00a9f4d381fee63359ded922775a342d',old_root=str(oldroot),changed_source_paths=changed,source_after=oldsources,products_after=oldproducts,scope='no production source differs; independently copied exact4e Cseed and complete ordinary providers; only scoped primitive/ownership claim'))
 query=report/'provider-query.mk';query.write_text('split_inventory:;@printf "%s\\n" "SPLIT_OBJECTS=$(COMPILER_OBJECTS)" "SPLIT_CFLAGS=$(CFLAGS)" "SPLIT_LDFLAGS=$(LDFLAGS)"'+'\n')
 raw,_,_=command('make-provider-inventory',['make','-f','Makefile.gnu','-f',query,'-s','split_inventory'])
 fields=dict(line.split('=',1) for line in raw.decode().splitlines() if line.startswith('SPLIT_'))
 objects=[(root/p).resolve() for p in shlex.split(fields['SPLIT_OBJECTS'])]
 dump('ordinary-provider-map.json',{str(p):identity(p) for p in objects})
 flags=fields['SPLIT_CFLAGS'];links=fields['SPLIT_LDFLAGS']
 if host=='puck':flags+=' '+shlex.join(['-isysroot',env['SDKROOT']]);links+=' '+shlex.join(['-isysroot',env['SDKROOT']])
 env.update(NANO_SPLIT_CC=cc,NANO_SPLIT_CFLAGS=flags,NANO_SPLIT_LDFLAGS=links,NANO_SPLIT_REPORT_DIR=str(report),NANO_SPLIT_COMMON_OBJECTS=shlex.join(str(p) for p in objects if p.name not in ('main.o','eval.o')),NANO_SPLIT_EVAL_OBJECT=str(root/'obj/eval.o'),NANO_SPLIT_CONTEXT_OBJECTS=shlex.join(str(p) for p in objects if p.name not in ('main.o','typechecker.o')),NANO_VISIBILITY_OBJECTS=shlex.join(str(p) for p in objects if p.name not in ('main.o','module.o')))
 manifest_data=dict(scope='Current4e Cseed only on Linux; two retained Nano roles qualified separately on Darwin; no full paired/SDK claim',source_pin=selected['pin'],compiler_pin='4e4a17be00a9f4d381fee63359ded922775a342d',selected_roles=['cseed'],producers={'cseed':dict(path=str(root/'bin/nanoc_c'),**identity(root/'bin/nanoc_c'))})
 dump('producers.json',manifest_data);env['NANO_SPLIT_PRODUCER_MANIFEST']=str(report/'producers.json')
 command('split-ownership',[sys.executable,'-m','unittest','-f','-v','tests.test_str_split_ownership'],1800)
 command('split-primitive',[sys.executable,'-m','unittest','-f','-v','tests.test_str_split_primitive'],1800)
 assert all(identity(oldroot/p)==value for p,value in oldproducts.items())
 dump('retained-products-after.json',{p:identity(oldroot/p) for p in oldproducts})

except BaseException as error:
 first=repr(error);dump('first-failure.json',dict(error=first));raise
finally:
 after=sources();ta=tools();dump('source-after.json',after);dump('tools-after.json',ta);dump('products-after.json',products());dump('terminal.json',dict(first_failure=first,source_equal=before==after,tools_equal=tb==ta));assert before==after and tb==ta
print('PASS',flush=True)
