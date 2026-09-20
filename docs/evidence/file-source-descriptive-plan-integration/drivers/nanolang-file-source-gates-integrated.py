import hashlib,json,os,pathlib,shlex,shutil,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]).resolve();report=pathlib.Path(sys.argv[2]).resolve();platform=sys.argv[3]
report.mkdir(parents=True,exist_ok=False);store=report/'objects';store.mkdir()
def dump(name,value): (report/name).write_text(json.dumps(value,indent=2)+'\n')
def hashfile(path,archive=False):
 p=pathlib.Path(path);h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data);size+=len(data)
 value={'sha256':h.hexdigest(),'bytes':size}
 if archive:
  dest=store/value['sha256']
  if not dest.exists():shutil.copyfile(p,dest)
  value['archive']=str(dest)
 return value
source_paths=json.loads(pathlib.Path(sys.argv[4]).read_text())
def sources():return {p:hashfile(root/p) for p in source_paths}
def products():
 paths=[]
 for base in (root/'bin',root/'obj',root/'modules/file_source_catalog/.build'):
  if base.exists():paths.extend(p for p in base.rglob('*') if p.is_file())
 return {str(p.relative_to(root)):hashfile(p,True) for p in sorted(paths)}
manifest=[]
def command(name,args,env,timeout=180):
 state={'argv':list(map(str,args)),'timeout':False,'returncode':None,'cleanup_errors':[]};proc=None;start=time.monotonic()
 dump(name+'-command.json',{'argv':state['argv'],'cwd':str(root),'environment':{k:env.get(k) for k in ('CC','SDKROOT','PATH','PKG_CONFIG_PATH','NMS_NATIVE_CLANG_FLAGS','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS','NANO_FILE_SOURCE_CC','NANO_FILE_SOURCE_CFLAGS','NANO_FILE_SOURCE_SANITIZERS')}})
 with (report/(name+'.stdout')).open('wb') as out,(report/(name+'.stderr')).open('wb') as err:
  try:
   proc=subprocess.Popen(state['argv'],cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
   try:state['returncode']=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:state['timeout']=True;state['returncode']=124
   dump(name+'-status.json',state)
  except OSError as e:state['os_error']=repr(e)
  finally:
   if proc is not None:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    except OSError as e:state['cleanup_errors'].append(repr(e))
    try:state['child_returncode']=proc.wait(timeout=10)
    except subprocess.TimeoutExpired:state['cleanup_errors'].append('post-kill wait timeout')
    end=time.monotonic()+5
    while True:
     try:os.killpg(proc.pid,0)
     except ProcessLookupError:state['group_absent']=True;break
     if time.monotonic()>=end:state['cleanup_errors'].append('remaining process group');break
     time.sleep(.02)
   state['seconds']=time.monotonic()-start;dump(name+'-status.json',state)
 manifest.append({'phase':name,**state});dump('manifest.json',manifest)
 print(name,state['returncode'],round(state['seconds'],3),flush=True)
 return state['returncode']==0 and not state['timeout'] and not state['cleanup_errors'] and state.get('group_absent')
env=dict(os.environ,LSAN_OPTIONS='',ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
if platform=='linux':
 cc='/usr/bin/gcc';other='/usr/local/bin/clang';extra=['--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13']
 env['NMS_NATIVE_CLANG_FLAGS']=extra[0]
 configs=[('gcc-ordinary',cc,[],False,True)]
else:
 cc='/usr/bin/clang';other='/opt/homebrew/opt/llvm/bin/clang';extra=[]
 env['PATH']='/opt/homebrew/bin:'+env.get('PATH','')
 configs=[('apple-ordinary',cc,[],False,True)]
make=shutil.which('make',path=env['PATH']);python=sys.executable
if platform!='linux':
 if not command('sdk',['/usr/bin/xcrun','--show-sdk-path'],env):sys.exit(1)
 env['SDKROOT']=(report/'sdk.stdout').read_text().strip()
 if not command('sdk-version',['/usr/bin/xcrun','--show-sdk-version'],env):sys.exit(1)
 env['PKG_CONFIG_PATH']='/opt/homebrew/opt/libffi/lib/pkgconfig:/opt/homebrew/opt/openssl@3/lib/pkgconfig'
env['CC']=cc
pkg=shutil.which('pkg-config',path=env['PATH'])
tools=[pathlib.Path(p).resolve() for p in (cc,other,python,make,pkg,shutil.which('ar',path=env['PATH']),shutil.which('ld',path=env['PATH'])) if p]
for name,args in [('host',['uname','-a']),('ordinary-version',[cc,'--version']),('other-version',[other,*extra,'--version']),('make-version',[make,'--version']),('pkg-config-version',[pkg,'--version']),('ffi-flags',[pkg,'--cflags','--libs','libffi'])]:
 if not command(name,args,env):sys.exit(1)
if platform!='linux':
 for path in (pathlib.Path('/opt/homebrew/opt/libffi/include/ffi.h'),pathlib.Path('/opt/homebrew/opt/libffi/include/ffitarget.h')):
  if path.is_file():tools.append(path.resolve())
 if not command('ffi-libdir',[pkg,'--variable=libdir','libffi'],env):sys.exit(1)
 libdir=pathlib.Path((report/'ffi-libdir.stdout').read_text().strip())
 tools.extend(p.resolve() for p in libdir.glob('libffi*.dylib'))
queries=[('other-asan',[other,*extra,'-print-file-name='+('libclang_rt.asan.a' if platform=='linux' else 'libclang_rt.asan_osx_dynamic.dylib')])]
if platform=='linux':queries += [('gcc-asan',[cc,'-print-file-name=libasan.so']),('gcc-ubsan',[cc,'-print-file-name=libubsan.so']),('gcc-cc1',[cc,'-print-prog-name=cc1']),('clang-ubsan',[other,*extra,'-print-file-name=libclang_rt.ubsan_standalone.a'])]
for name,args in queries:
 if not command(name,args,env):sys.exit(1)
 p=pathlib.Path((report/(name+'.stdout')).read_text().strip())
 if not p.is_file():raise RuntimeError('Unresolved actual tool '+str(p))
 tools.append(p.resolve())
def toolmap():return {str(p):hashfile(p) for p in sorted(set(tools))}
dump('driver.json',hashfile(__file__,True));dump('configuration.json',{'pin':'af68d82ae','platform':platform,'configs':configs,'source_count':len(source_paths),'source_scope':'all tracked files at the frozen pin','sanitizer_scope':'ordinary integration only; original sanitizer acceptance remains at701','tool_scope':'explicit drivers, selected subtools, sanitizer libraries and Darwin ffi headers/library; not complete transitive system toolchain'})
def phase(name,args,phase_env,timeout):
 before=sources();tb=toolmap();dump(name+'-source-before.json',before);dump(name+'-tools-before.json',tb);dump(name+'-products-before.json',products())
 ok=command(name,args,phase_env,timeout)
 after=sources();ta=toolmap();dump(name+'-source-after.json',after);dump(name+'-tools-after.json',ta);dump(name+'-products-after.json',products())
 artifacts={str(p.relative_to(report)):hashfile(p,True) for p in sorted(report.rglob('*')) if p.is_file() and store not in p.parents and p.suffix not in ('.json',)}
 dump(name+'-artifacts.json',artifacts)
 if before!=after or tb!=ta:dump(name+'-changed-inputs.json',{'source_equal':before==after,'tools_equal':tb==ta});return False
 return ok
old=pathlib.Path(sys.argv[5]); reused={}
for name in ('nanoc','nanoc_stage1','nanoc_stage2'):
 source=old/'bin'/name; dest=root/'bin'/name; dest.parent.mkdir(exist_ok=True)
 if dest.exists():raise RuntimeError('Existing compiler '+str(dest))
 shutil.copy2(source,dest);reused[name]={'source':str(source),'original':hashfile(source),'copied':hashfile(dest,True)}
 assert reused[name]['original']['sha256']==reused[name]['copied']['sha256']
for name in source_paths:
 if name.startswith('src_nano/'):
  assert hashfile(old/name)==hashfile(root/name),name
for bad in (root/'obj',root/'modules/file_source_catalog/.build'):
 if bad.exists():raise RuntimeError('Not fresh '+str(bad))
dump('bootstrap-reuse.json',{'original_bootstrap':'32ade5b5d','retained_qualified_tree':str(old),'copied':reused,'all_src_nano_exact':True,'Cseed_and_all_C_providers':'fresh build at integration pin; no reused objects'})
if platform!='linux':
 if not command('owning-apple-clang',['/usr/bin/xcrun','--find','clang'],env):sys.exit(1)
 actual=pathlib.Path((report/'owning-apple-clang.stdout').read_text().strip());tools.append(actual.resolve())
 if not command('owning-apple-version',[str(actual),'--version'],env):sys.exit(1)
if not phase('fresh-cseed-providers',[make,'-f','Makefile.gnu','-j2','CC='+cc,'bin/nanoc_c'],env,1800):sys.exit(1)

for label,compiler,flags,sanitizers,paired in configs:
 where=report/label;where.mkdir()
 selected=shlex.join([compiler,*flags])
 phase_env=dict(env,CC=selected,NANO_FILE_SOURCE_CC=selected,NANO_FILE_SOURCE_CFLAGS='',NANO_FILE_SOURCE_SANITIZERS='1' if sanitizers else '0',NANO_FILE_SOURCE_REPORT_DIR=str(where))
 target='tests.test_file_source_plan' if paired else 'tests.test_file_source_plan.FileSourcePlan.test_c_ownership_allocation_and_exact_budget'
 if not phase(label,[python,'-m','unittest','-f','-v',target],phase_env,1800):sys.exit(1)
dump('terminal.json',{'status':'PASS','pin':'af68d82ae','configs':[x[0] for x in configs],'all_input_pairs_equal':True})
