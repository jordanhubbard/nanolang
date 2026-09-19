import hashlib,json,pathlib,subprocess,os,time,shutil,sys,re
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);platform=sys.argv[3];out.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','modules','stdlib','Makefile','Makefile.gnu').splitlines()
def source():return {p:sha(root/p) for p in files if (root/p).is_file()}
tools=[pathlib.Path(shutil.which(p)).resolve() for p in ['python3','make','gcc','clang','cc','as','ld'] if shutil.which(p)]
if platform=='darwin':
 tools.append(pathlib.Path('/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'))
 sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip();os.environ['SDKROOT']=sdk
 write('sdk.json',{'path':sdk,'settings_sha256':sha(pathlib.Path(sdk)/'SDKSettings.json'),'xcrun_sha256':sha(pathlib.Path('/usr/bin/xcrun'))})
def toolmap():return {str(p):sha(p) for p in tools}
before=source();tb=toolmap();write('source-before.json',before);write('tools-before.json',tb);shutil.copyfile(__file__,out/'runner.py')
report={'pin':git('rev-parse','HEAD'),'platform':platform,'steps':[]};status=0
normal=[] if platform=='linux' else ['CC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang']
steps=[('normal',['make','-j4','test-file-nominal',*normal],{})]
if platform=='linux':steps += [('gcc',['make','test-file-nominal-sanitizers'],{'NANO_FILE_NOMINAL_CC':'/bin/gcc'}),('clang',['make','test-file-nominal-sanitizers'],{'NANO_FILE_NOMINAL_CC':'/usr/local/bin/clang','NANO_FILE_NOMINAL_CFLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'})]
else:steps += [('clang',['make','test-file-nominal-sanitizers'],{'NANO_FILE_NOMINAL_CC':'/opt/homebrew/opt/llvm/bin/clang'})]
steps += [('adjacent',['make','test-service-bindings','test-nsi-file-plan',*normal],{})]
try:
 for name,cmd,env in steps:
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as f:r=subprocess.run(cmd,cwd=root,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT)
  status=r.returncode;report['steps'].append({'name':name,'command':cmd,'env':{**env,**({'SDKROOT':os.environ['SDKROOT']} if platform=='darwin' else {})},'status':status,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(out/(name+'.log'))});write('manifest.json',report)
  a=out/(name+'-artifacts');a.mkdir();mapping={}
  for p in (root/'obj').rglob('*'):
   if p.is_file():
    rel=p.relative_to(root/'obj');q=a/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);mapping[str(p)]=sha(p)
  for location in re.findall(r'artifacts at (\S+)',(out/(name+'.log')).read_text()):
   for p in pathlib.Path(location).iterdir():
    if p.is_file():shutil.copyfile(p,a/p.name);mapping[str(p)]=sha(p)
  write(name+'-artifacts.json',mapping);print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=source();ta=toolmap();write('source-after.json',after);write('tools-after.json',ta);report.update(terminal=status,source_equal=before==after,tools_equal=tb==ta,tracked_clean=not git('status','--porcelain','--untracked-files=no'));write('manifest.json',report)
raise SystemExit(status)
