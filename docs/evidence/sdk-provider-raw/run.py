import pathlib,hashlib,json,platform,os,sys,subprocess,signal,time,tarfile
base=pathlib.Path(sys.argv[1]);base.mkdir();archive=pathlib.Path(sys.argv[2]);root=base/'source';root.mkdir()
with tarfile.open(archive) as tf:tf.extractall(root)
def ident(p):return {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'mode':p.stat().st_mode&511}
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
files={str(p.relative_to(root)):ident(p) for p in root.rglob('*') if p.is_file()};save(base/'source-before.json',files)
darwin=platform.system()=='Darwin';sdk=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip() if darwin else ''
compilers=['/usr/bin/clang','/opt/homebrew/opt/llvm/bin/clang'] if darwin else ['/usr/bin/gcc','/usr/bin/gcc'];tools={c:ident(pathlib.Path(c).resolve()) for c in compilers};save(base/'identity.json',{'pin':'df714877f','archive':ident(archive),'driver':ident(pathlib.Path(__file__)),'host':platform.uname()._asdict(),'sdk':sdk,'tools':tools})
env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1');env.pop('LSAN_OPTIONS',None);records=[]
def run(label,args):
 start=time.monotonic();record={'label':label,'argv':args,'bound':60,'timeout':False};p=None
 try:
  with (base/(label+'.log')).open('wb') as log:
   p=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=p.pid
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:record['timeout']=True;os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
 finally:
  record['seconds']=time.monotonic()-start;record['returncode']=p.returncode if p else None;record['group_remaining']=False
  if p:
   try:os.killpg(p.pid,0);record['group_remaining']=True;os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  records.append(record);save(base/'phases.json',records)
 print(record,flush=True);assert record['returncode']==0 and not record['timeout'] and not record['group_remaining']
try:
 for mode,cc in zip(['ordinary','sanitized'],compilers):
  flags=['-std=c99','-Wall','-Wextra','-Werror','-g','-O1' if mode=='sanitized' else '-O2']
  if darwin:flags+=['-isysroot',sdk]
  if mode=='sanitized':flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-sanitize-recover=all']
  exe=base/(mode+'-codec');run(mode+'-build',[cc,*flags,'tests/test_sdk_provider_codec.c','-o',str(exe)]);run(mode+'-codec',[str(exe)])
  save(base/(mode+'-product.json'),ident(exe))
 after={n:ident(root/n) for n in files};save(base/'source-after.json',after);assert after==files
 ta={c:ident(pathlib.Path(c).resolve()) for c in compilers};save(base/'tools-after.json',ta);assert ta==tools;save(base/'terminal.json',{'status':0})
except BaseException as e:
 save(base/'source-after-failure.json',{n:ident(root/n) for n in files});save(base/'terminal.json',{'status':1,'error':repr(e)});raise
