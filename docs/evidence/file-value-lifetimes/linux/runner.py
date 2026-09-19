import hashlib,json,pathlib,subprocess,os,time,shutil,sys
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);platform=sys.argv[3];out.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','modules','stdlib','Makefile','Makefile.gnu').splitlines()
def source():return {p:sha(root/p) for p in files if (root/p).is_file()}
tools=[pathlib.Path(shutil.which(p)).resolve() for p in ['python3','make','gcc','clang','cc','as','ld'] if shutil.which(p)]
if platform=='darwin':tools.append(pathlib.Path('/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'))
def toolmap():return {str(p):sha(p) for p in tools}
def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
before=source();tb=toolmap();write('source-before.json',before);write('tools-before.json',tb);shutil.copyfile(__file__,out/'runner.py')
report={'pin':git('rev-parse','HEAD'),'platform':platform,'steps':[]};status=0
if platform=='linux':
 steps=[('normal',['make','test-nsi-file-values'],{}),('gcc',['make','test-nsi-file-values-sanitizers'],{'NANO_FILE_VALUES_CC':'/bin/gcc'}),('clang',['make','test-nsi-file-values-sanitizers'],{'NANO_FILE_VALUES_CC':'/usr/local/bin/clang','NANO_FILE_VALUES_CFLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'}),('adjacent',['make','test-nsi-file','test-nsi-cap','test-nsi-file-plan'],{})]
else:
 steps=[('normal',['make','test-nsi-file-values','CC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'],{}),('clang',['make','test-nsi-file-values-sanitizers'],{'NANO_FILE_VALUES_CC':'/opt/homebrew/opt/llvm/bin/clang'}),('adjacent',['make','test-nsi-file','test-nsi-cap','test-nsi-file-plan','CC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'],{})]
try:
 for name,cmd,env in steps:
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as f:r=subprocess.run(cmd,cwd=root,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT)
  status=r.returncode;report['steps'].append({'name':name,'command':cmd,'env':env,'status':status,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(out/(name+'.log'))});write('manifest.json',report)
  archive=out/(name+'-artifacts');archive.mkdir();mapping={}
  for p in (root/'obj').glob('test_nsi_file_values*'):
   if p.is_file():shutil.copyfile(p,archive/p.name);mapping[str(p)]=sha(p)
  import re
  for location in re.findall(r'artifacts at (\S+)',(out/(name+'.log')).read_text()):
   for p in pathlib.Path(location).iterdir():
    if p.is_file():shutil.copyfile(p,archive/p.name);mapping[str(p)]=sha(p)
  write(name+'-artifacts.json',mapping)
  print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=source();ta=toolmap();write('source-after.json',after);write('tools-after.json',ta);report.update(terminal=status,source_equal=before==after,tools_equal=tb==ta,tracked_clean=not git('status','--porcelain','--untracked-files=no'));write('manifest.json',report)
raise SystemExit(status)
