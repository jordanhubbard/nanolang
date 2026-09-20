import os,sys,json,hashlib,pathlib,subprocess,time,signal,shutil,re
root=pathlib.Path('/home/jkh/Src/nanolang-opencl-unload-mapping')
out=pathlib.Path('/tmp/nanolang-opencl-unload-4b71');out.mkdir(exist_ok=False)
os.chdir(root)
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
def cmd(a):return subprocess.check_output(a,text=True).strip()
def sources():return {p:digest(root/p) for p in cmd(['git','ls-files']).splitlines() if (root/p).is_file()}
def inventory(paths):return {str(p):{'realpath':str(pathlib.Path(p).resolve()),'sha256':digest(p)} for p in sorted(set(paths))}
def phase(name,argv):
 start=time.monotonic();env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'}
 with (out/(name+'.log')).open('w') as f:
  p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:rc=p.wait(timeout=120)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   rc='timeout'
 row={'argv':argv,'cwd':str(root),'exit_code':rc,'seconds':time.monotonic()-start,'ASAN_OPTIONS':env['ASAN_OPTIONS'],'UBSAN_OPTIONS':env['UBSAN_OPTIONS'],'LSAN_OPTIONS':env.get('LSAN_OPTIONS')};save(name+'.json',row);return rc
head=cmd(['git','rev-parse','HEAD']);assert head.startswith('4b71cc2ab');assert not cmd(['git','status','--porcelain'])
original=json.load(open('/tmp/nanolang-opencl-qualified/sources-before.json'))
for p in ['tests/test_opencl_identity_real.c','modules/gpu/opencl_runtime.c']:
 assert digest(root/p)==original[p]
cc='/usr/bin/aarch64-linux-gnu-gcc-13';toolpaths=[cc,cmd([cc,'-print-prog-name=cc1']),'/usr/bin/as','/usr/bin/ld',sys.executable,'/usr/bin/ldd']
libpaths=['/usr/lib/aarch64-linux-gnu/libOpenCL.so.1','/usr/lib/aarch64-linux-gnu/libnvidia-opencl.so.1','/etc/OpenCL/vendors/nvidia.icd',cmd([cc,'-print-file-name=libasan.so']),cmd([cc,'-print-file-name=libubsan.so'])]
for p in libpaths[:2]+libpaths[3:]:
 text=cmd(['/usr/bin/ldd',p]);(out/('ldd-'+pathlib.Path(p).name+'.log')).write_text(text+'\n')
 for line in text.splitlines():
  for item in line.split():
   if item.startswith('/') and pathlib.Path(item).is_file():libpaths.append(item)
before=sources();tools=inventory(toolpaths);libs=inventory(libpaths)
save('source-before.json',before);save('tools-before.json',tools);save('libraries-before.json',libs);shutil.copy2(__file__,out/'runner.py')
status={'head':head,'scope':'Single diagnostic; original GPU assertions retained. No release or sanitizer acceptance inferred.'}
try:
 binary=str(out/'real-mapped');argv=[cc,'-std=c11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-omit-frame-pointer','tests/test_opencl_identity_real.c','tests/diagnostics/opencl_unload_maps.c','-Wl,--wrap=dlclose','-ldl','-o',binary]
 status['build_exit']=phase('build',argv)
 if status['build_exit']==0:
  save('binary.json',{'path':binary,'sha256':digest(binary)})
  status['run_exit']=phase('run',[binary])
finally:
 after=sources();ta=inventory(toolpaths);la=inventory(libpaths)
 save('source-after.json',after);save('tools-after.json',ta);save('libraries-after.json',la)
 status.update(sources_unchanged=before==after,tools_unchanged=tools==ta,libraries_unchanged=libs==la,head_unchanged=cmd(['git','rev-parse','HEAD'])==head)
 save('status.json',status);print(json.dumps(status))
