import hashlib,json,pathlib,subprocess,os,time,shutil,sys
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','modules','stdlib','Makefile','Makefile.gnu').splitlines()
def source():return {p:sha(root/p) for p in files if (root/p).is_file()}
tools=[pathlib.Path(shutil.which(p)).resolve() for p in ['python3','make','gcc','clang','as','ld','llvm-as','opt','llc'] if shutil.which(p)]
tools.append(pathlib.Path('/tmp/nanolang-service-integration-clang'))
def toolmap():return {str(p):sha(p) for p in tools}
def generated():return {str(p.relative_to(root)):sha(p) for p in (root/'bin').glob('*') if p.is_file()}
def write(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
before=source();tb=toolmap();write('source-before.json',before);write('tools-before.json',tb);shutil.copyfile(__file__,out/'runner.py')
report={'pin':git('rev-parse','HEAD'),'bootstrap_reference':'c34c7324f48522656c27f9cdd84489634634bd81','NMS_NATIVE_CLANG_FLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13','steps':[]};status=0
steps=[('gcc',['make','-j8','test-service-module'],{'NANO_SERVICE_MODULE_TEST_CC':'/bin/gcc'}),
 ('clang',['make','test-service-module'],{'NANO_SERVICE_MODULE_TEST_CC':'/usr/local/bin/clang','NANO_SERVICE_MODULE_TEST_CFLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'}),
 ('mixed-setup',['make','-j8','mixed-samples-runtime-fixture'],{}),
 ('mixed-gcc',['python3','-m','unittest','-fv','tests.test_mixed_samples_runtime'],{'CC':'/bin/gcc','NANO_MIXED_RUNTIME_DIR':str(out/'mixed-gcc-artifacts')}),
 ('mixed-clang',['python3','-m','unittest','-fv','tests.test_mixed_samples_runtime'],{'CC':'/tmp/nanolang-service-integration-clang','NANO_MIXED_RUNTIME_DIR':str(out/'mixed-clang-artifacts')}),
 ('adjacent',['make','test-mixed-samples-runtime-alloc','test-mixed-samples-admission','test-nvm-v2-convert','test-nvm-v2-module','test-verifier-profiles'],{})]
for name in ['mixed-gcc-artifacts','mixed-clang-artifacts']:(out/name).mkdir()

try:
 for name,cmd,env in steps:
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as f:r=subprocess.run(cmd,cwd=root,env={**os.environ,'NMS_NATIVE_CLANG_FLAGS':'--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',**env},stdout=f,stderr=subprocess.STDOUT)
  status=r.returncode;report['steps'].append({'name':name,'command':cmd,'env':env,'status':status,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(out/(name+'.log'))});write('manifest.json',report);write(name+'-generated-tools.json',generated());archive=out/(name+'-tools');archive.mkdir();[shutil.copyfile(p,archive/p.name) for p in (root/'bin').glob('*') if p.is_file()];print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=source();ta=toolmap();write('source-after.json',after);write('tools-after.json',ta);report.update(terminal=status,source_equal=before==after,tools_equal=tb==ta,tracked_clean=not git('status','--porcelain','--untracked-files=no'));write('manifest.json',report)
raise SystemExit(status)
