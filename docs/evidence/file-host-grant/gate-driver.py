import os,sys,json,hashlib,subprocess,time,pathlib,shutil,signal,platform
root=pathlib.Path(sys.argv[1]).resolve(); report=pathlib.Path(sys.argv[2]).resolve(); report.mkdir(parents=True,exist_ok=False)
tracked=json.loads(pathlib.Path(sys.argv[3]).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def sources():return {p:sha(root/p) for p in tracked}
def capture(cmd):return subprocess.check_output(cmd,text=True).strip()
darwin=platform.system()=='Darwin'
make=shutil.which('make');python=sys.executable
sdk=capture(['xcrun','--sdk','macosx','--show-sdk-path']) if darwin else None
apple=capture(['xcrun','--find','clang']) if darwin else None
configs=[('apple-normal',apple,False),('homebrew-normal','/opt/homebrew/opt/llvm/bin/clang',False),('homebrew-sanitizers','/opt/homebrew/opt/llvm/bin/clang',True)] if darwin else [('gcc-normal',shutil.which('gcc'),False),('clang-normal',shutil.which('clang'),False),('gcc-sanitizers',shutil.which('gcc'),True),('clang-sanitizers',shutil.which('clang'),True)]
write(report/'source-before.json',sources())
write(report/'host.json',{'platform':platform.platform(),'machine':platform.machine(),'source_pin':'54822f541','source_root':str(root),'sdk':sdk,'selected_source_count':len(tracked),'scope':'grant only; no service/dispatcher execution'})
alltools={}; terminals=[]
for name,cc,san in configs:
 phase=report/name;phase.mkdir();(phase/'fixtures').mkdir()
 env=dict(os.environ);env.pop('LSAN_OPTIONS',None);env['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1';env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1';env['TMPDIR']=str(phase/'fixtures');env['PYTHONDONTWRITEBYTECODE']='1'
 if sdk:env['SDKROOT']=sdk
 flags=['-O1','-g','-Wall','-Wextra','-Werror','-std=c99','-fPIC']
 if sdk:flags+=['-isysroot',sdk]
 if not darwin and 'clang' in cc:flags+=['--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13']
 if san:flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
 tools={'compiler':cc,'make':make,'python':python}
 for prog in ['cc1','as','ld']:
  candidate=capture([cc,'-print-prog-name='+prog]);p=shutil.which(candidate) if not os.path.isabs(candidate) else candidate
  if p and os.path.isfile(p):tools[prog]=p
 tools={k:{'path':str(pathlib.Path(v).resolve()),'sha256':sha(pathlib.Path(v).resolve())} for k,v in tools.items()}
 write(phase/'tools-before.json',tools);alltools[name]=tools
 write(phase/'compiler.json',{'selected':cc,'version':capture([cc,'--version']),'flags':flags,'sanitizers':san,'env':{k:env.get(k) for k in ['SDKROOT','LSAN_OPTIONS','ASAN_OPTIONS','UBSAN_OPTIONS']}})
 def run(label,cmd):
  start=time.monotonic();proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
  try:output,_=proc.communicate(timeout=180)
  except subprocess.TimeoutExpired:
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   output,_=proc.communicate();status='timeout'
  else:status=proc.returncode
  (phase/(label+'.log')).write_bytes(output)
  result={'command':cmd,'status':status,'elapsed_seconds':round(time.monotonic()-start,3)}
  write(phase/(label+'.json'),result);terminals.append({'phase':name,'label':label,**result});write(report/'terminals.json',terminals)
  print(name,label,status,result['elapsed_seconds'],flush=True)
  return status==0
 # Explicit Make recipe builds a never-before-existing per-phase object.
 obj=phase/'obj';cmd=[make,'-f','Makefile.gnu','file-host-grant','CC='+cc,'CFLAGS='+' '.join(flags),'OBJ_DIR='+str(obj),'LIBFFI_CFLAGS=','LIBFFI_LIBS=']
 good=run('make',cmd)
 provider=obj/'nanoisa/file_host_grant.o'
 if good:
  env.update(NANO_FILE_HOST_GRANT_CC=cc,NANO_FILE_HOST_GRANT_CFLAGS=' '.join(flags),NANO_FILE_HOST_GRANT_CPPFLAGS='',NANO_FILE_HOST_GRANT_LDFLAGS=' '.join(x for x in flags if x.startswith('-fsanitize=')),NANO_FILE_HOST_GRANT_OBJECT=str(provider))
  good=run('fixture',[python,'-m','unittest','-f','-v','tests.test_file_host_grant'])
 # Actual transitive system header identity, separate from project source map.
 if good:
  deps=subprocess.check_output([cc]+flags+['-std=c11','-M','-Isrc/nanoisa','src/nanoisa/file_host_grant.c'],cwd=root,env=env,text=True)
  (phase/'headers.d').write_text(deps)
  tokens=deps.replace('\\\n',' ').split()[1:]
  headers={}
  for p in tokens:
   path=pathlib.Path(p);path=path if path.is_absolute() else root/path
   if path.is_file():headers[str(path.resolve())]=sha(path)
  write(phase/'headers.json',headers)
 after={k:{'path':v['path'],'sha256':sha(v['path'])} for k,v in tools.items()};write(phase/'tools-after.json',after)
 if after!=tools:raise RuntimeError('tool changed')
 artifacts={str(p.relative_to(phase)):sha(p) for p in phase.rglob('*') if p.is_file() and p.suffix in ['.o','.d']}
 for p in (phase/'fixtures').rglob('*'):
  if p.is_file() and p.name in ['linked','instrumented']:artifacts[str(p.relative_to(phase))]=sha(p)
 write(phase/'artifacts.json',artifacts)
 if not good:break
write(report/'source-after.json',sources())
if json.loads((report/'source-before.json').read_text())!=json.loads((report/'source-after.json').read_text()):raise RuntimeError('source changed')
write(report/'summary.json',{'pass':good and len(terminals)==len(configs)*2,'terminal_count':len(terminals),'source_count':len(tracked),'configurations':len(configs),'source_pin':'54822f541'})
sys.exit(0 if good and len(terminals)==len(configs)*2 else 1)
