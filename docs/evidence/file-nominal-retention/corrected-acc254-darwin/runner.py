import hashlib,json,pathlib,subprocess,os,time,shutil,sys,re
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);platform=sys.argv[3];out.mkdir();store=out/'artifact-store';store.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','modules','stdlib','Makefile','Makefile.gnu').splitlines()
def source():return {p:sha(root/p) for p in files if (root/p).is_file()}
tools=[pathlib.Path(shutil.which(p)).resolve() for p in ['python3','make','gcc','clang','cc','as','ld','llvm-as','opt','llc'] if shutil.which(p)]
baseenv={}
if platform=='darwin':
 tools.append(pathlib.Path('/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'))
 baseenv['CC']='/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'
 sdk=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip();baseenv['SDKROOT']=sdk
 write('sdk.json',{'path':sdk,'settings_sha256':sha(pathlib.Path(sdk)/'SDKSettings.json'),'xcrun_sha256':sha(pathlib.Path('/usr/bin/xcrun'))})
else:baseenv['NMS_NATIVE_CLANG_FLAGS']='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'
def toolmap():return {str(p):sha(p) for p in tools}
def bins():return {str(p.relative_to(root)):sha(p) for p in (root/'bin').glob('*') if p.is_file()}
def archive(paths):
 result={}
 for p in paths:
  if not p.is_file():continue
  h=sha(p);target=store/h
  if not target.exists():shutil.copyfile(p,target)
  assert sha(target)==h;result[str(p)]={'sha256':h,'bytes':p.stat().st_size}
 return result
before=source();tb=toolmap();write('source-before.json',before);write('tools-before.json',tb);shutil.copyfile(__file__,out/'runner.py')
report={'pin':git('rev-parse','HEAD'),'platform':platform,'bootstrap_reference':'f5e07e5bea043265cfc15176fb1804227c363901','steps':[]};status=0
steps=[('setup',['make','-j8','nano_virt','nano_vm','nvm2c','nvm2llvm','nvm2hl'],{})]
if platform=='linux':
 steps += [('gcc',['make','test-file-nominal-module','test-file-nominal','test-file-nominal-sanitizers','test-service-module'],{'NANO_SERVICE_MODULE_TEST_CC':'/bin/gcc','NANO_FILE_NOMINAL_CC':'/bin/gcc'}),('clang',['make','test-file-nominal-module','test-file-nominal-sanitizers','test-service-module'],{'NANO_SERVICE_MODULE_TEST_CC':'/usr/local/bin/clang','NANO_SERVICE_MODULE_TEST_CFLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13','NANO_FILE_NOMINAL_CC':'/usr/local/bin/clang','NANO_FILE_NOMINAL_CFLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'})]
else:steps += [('darwin',['make','test-file-nominal-module','test-file-nominal','test-file-nominal-sanitizers','test-service-module'],{'NANO_SERVICE_MODULE_TEST_CC':'/opt/homebrew/opt/llvm/bin/clang','NANO_FILE_NOMINAL_CC':'/opt/homebrew/opt/llvm/bin/clang'})]
steps += [('adjacent',['make','test-service-bindings','test-nsi-file-plan','test-nvm-format-v2','test-nvm-v2-imports','test-nvm-v2-module','test-nvm-v2-convert','test-nvm-v2-endtoend','test-wrapper-gen','test-verifier-profiles','test-owned-array-layouts','test-owned-array-origins'],{})]
try:
 for name,cmd,env in steps:
  print('START',name,flush=True);start=time.monotonic();write(name+'-bins-before.json',bins())
  with (out/(name+'.log')).open('wb') as f:r=subprocess.run(cmd,cwd=root,env={**os.environ,**baseenv,**env},stdout=f,stderr=subprocess.STDOUT)
  status=r.returncode;report['steps'].append({'name':name,'command':cmd,'env':{**baseenv,**env},'status':status,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(out/(name+'.log'))});write('manifest.json',report)
  paths=list((root/'bin').glob('*'))+list((root/'obj').glob('test_*'))
  for location in re.findall(r'artifacts at (\S+)',(out/(name+'.log')).read_text()):paths+=list(pathlib.Path(location).rglob('*'))
  write(name+'-artifacts.json',archive(paths));write(name+'-bins-after.json',bins());write(name+'-objects.json',{str(p.relative_to(root)):sha(p) for p in (root/'obj').rglob('*.o')});print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=source();ta=toolmap();write('source-after.json',after);write('tools-after.json',ta);write('bins-final.json',bins());report.update(terminal=status,source_equal=before==after,tools_equal=tb==ta,tracked_clean=not git('status','--porcelain','--untracked-files=no'));write('manifest.json',report)
raise SystemExit(status)
