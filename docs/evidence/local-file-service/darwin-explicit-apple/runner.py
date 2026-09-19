import hashlib,json,pathlib,subprocess,os,time,shutil,sys
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','modules','stdlib','Makefile','Makefile.gnu').splitlines()
def source():return {p:sha(root/p) for p in files if (root/p).is_file()}
ccs=[('gcc','/bin/gcc',{}),('clang','/usr/local/bin/clang',{'NANO_FILE_TEST_CFLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'})] if sys.platform!='darwin' else [('darwin','/opt/homebrew/opt/llvm/bin/clang',{})]
toolpaths=[pathlib.Path('/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang')] + [pathlib.Path(shutil.which(p)).resolve() for p in ['python3','make','as','ld'] if shutil.which(p)]+[pathlib.Path(c).resolve() for _,c,_ in ccs]
def tools():return {str(p):sha(p) for p in toolpaths}
def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
before=source();tb=tools();write('source-before.json',before);write('tools-before.json',tb);shutil.copyfile(__file__,out/'runner.py')
report={'pin':git('rev-parse','HEAD'),'steps':[]};status=0
steps=[('adjacent-explicit-apple',['make','CC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang','test-nsi-cap','test-nsi-shm','test-nsi-fabric','test-nsi-runtime','test-nsi-gen','test-nsi'],{})]
try:
 for name,cmd,env in steps:
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as f:r=subprocess.run(cmd,cwd=root,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT)
  status=r.returncode;report['steps'].append({'name':name,'command':cmd,'env':env,'status':status,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(out/(name+'.log'))});write('manifest.json',report);print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=source();ta=tools();write('source-after.json',after);write('tools-after.json',ta);report.update(terminal=status,source_equal=before==after,tools_equal=tb==ta,tracked_clean=not git('status','--porcelain','--untracked-files=no'));write('manifest.json',report)
raise SystemExit(status)
