import pathlib,subprocess,os,sys,json,hashlib,time,signal,shutil,tarfile
root=pathlib.Path('/home/jkh/Src/nanolang-opencl-private-loader');out=pathlib.Path('/tmp/nanolang-opencl-private-2397');out.mkdir(exist_ok=False)
source_archive=pathlib.Path('/tmp/nanolang-ocl-icd-symbol-audit/upstream-3e1155.tar.gz');toolsroot=pathlib.Path('/tmp/nanolang-opencl-loader-tools');prefix=out/'install';build=out/'upstream'
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
def check(a,cwd=None):return subprocess.check_output(a,cwd=cwd,text=True).strip()
def filesmap(p):return {str(f.relative_to(p)):digest(f) for f in sorted(p.rglob('*')) if f.is_file()}
def inventory(paths):return {str(p):{'realpath':str(pathlib.Path(p).resolve()),'sha256':digest(p)} for p in sorted(set(paths),key=str)}
def sources():return {p:digest(root/p) for p in check(['git','ls-files'],root).splitlines() if (root/p).is_file()}
head=check(['git','rev-parse','HEAD'],root);assert head.startswith('2397da09e');assert not check(['git','status','--porcelain'],root)
assert digest(source_archive)=='c94648dbe9d2b3f72a9e2710406ac295fd729787cc4ab715bec1993bf996478a'
package=pathlib.Path('/tmp/libtool_2.4.7-7build1_all.deb');meta=check(['/usr/bin/apt-cache','show','libtool=2.4.7-7build1']);(out/'libtool-package-metadata.txt').write_text(meta+'\n');expected=next(x.split(': ',1)[1] for x in meta.splitlines() if x.startswith('SHA256: '));assert digest(package)==expected
save('source-package.json',{'url':'https://codeload.github.com/OCL-dev/ocl-icd/tar.gz/3e1155f0796cb9ac0fe309664d98e4b0ed2c3300','sha256':digest(source_archive),'libtool_deb_sha256':expected})
shutil.copytree('/tmp/nanolang-ocl-icd-symbol-audit/ocl-icd-3e1155f0796cb9ac0fe309664d98e4b0ed2c3300',build)
data=out/'libtool-data';data.mkdir()
for name,target in [('build-aux',toolsroot/'usr/share/libtool/build-aux'),('m4',toolsroot/'usr/share/aclocal'),('libltdl',pathlib.Path('/usr/share/libtool'))]:(data/name).symlink_to(target,target_is_directory=True)
ruby=pathlib.Path('/home/jkh/Src/RubyOS/build/host-ruby/bin/ruby');cc='/usr/bin/aarch64-linux-gnu-gcc-13';clang='/usr/bin/clang-18'
env={**os.environ,'PATH':str(toolsroot/'usr/bin')+':'+str(ruby.parent)+':/usr/bin:/bin','_lt_pkgdatadir':str(data),'ACLOCAL_PATH':str(toolsroot/'usr/share/aclocal'),'CC':cc,'CFLAGS':'-O2 -g'}
controlnames=['OCL_ICD_FORCE_LEGACY_TERMINATION','OCL_ICD_DISABLE_DYNAMIC_LIBRARY_UNLOADING','OCL_ICD_VENDORS','OCL_ICD_FILENAMES','OPENCL_VENDOR_PATH','OPENCL_LAYERS','OPENCL_LAYER_PATH','OCL_ICD_LAYERS','OCL_ICD_DEBUG','LD_PRELOAD']
for key in controlnames:env.pop(key,None)
env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='')
save('environment.json',{k:env.get(k) for k in controlnames+['PATH','_lt_pkgdatadir','ACLOCAL_PATH','CC','CFLAGS','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS']})
toolpaths=[cc,clang,str(ruby),sys.executable,'/usr/bin/as','/usr/bin/ld','/usr/bin/make','/usr/bin/m4','/usr/bin/perl','/usr/bin/autoreconf','/usr/bin/autoconf','/usr/bin/automake','/usr/bin/aclocal','/usr/bin/autoheader',str(toolsroot/'usr/bin/libtoolize'),check([cc,'-print-prog-name=cc1'])]
for directory in [toolsroot/'usr/share',pathlib.Path('/usr/share/autoconf'),pathlib.Path('/usr/share/automake-1.16'),pathlib.Path('/usr/share/aclocal-1.16'),pathlib.Path('/usr/share/libtool'),ruby.parent.parent/'lib/ruby']:
 if directory.exists():toolpaths.extend(str(p) for p in directory.rglob('*') if p.is_file())
libpaths=['/usr/lib/aarch64-linux-gnu/libOpenCL.so.1','/usr/lib/aarch64-linux-gnu/libnvidia-opencl.so.1','/etc/OpenCL/vendors/nvidia.icd']
for pattern in ['/etc/OpenCL/vendors']:
 libpaths.extend(str(p) for p in pathlib.Path(pattern).glob('*') if p.is_file())
before=sources();tb=inventory(toolpaths);lb=inventory(libpaths);upstreambefore=filesmap(build)
save('source-before.json',before);save('tools-before.json',tb);save('system-libraries-before.json',lb);save('upstream-originals.json',upstreambefore);shutil.copy2(__file__,out/'runner.py')
rows=[]
def phase(name,argv,cwd,bound=600,extra=None):
 effective={**env,**(extra or {})};start=time.monotonic()
 with (out/(name+'.log')).open('w') as f:
  p=subprocess.Popen(argv,cwd=cwd,env=effective,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=bound)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   rc='timeout'
 row={'name':name,'argv':argv,'cwd':str(cwd),'exit_code':rc,'seconds':time.monotonic()-start,'LD_LIBRARY_PATH':effective.get('LD_LIBRARY_PATH')};rows.append(row);save(name+'.json',row);print(name,rc,flush=True)
 if rc!=0:raise RuntimeError(name+' first terminal '+str(rc))
status={'head':head,'success':False}
try:
 phase('bootstrap',['./bootstrap'],build)
 phase('configure',['./configure','--prefix='+str(prefix),'--enable-official-khronos-headers','--disable-update-database'],build)
 phase('make',['/usr/bin/make','-j2'],build)
 phase('install-private',['/usr/bin/make','install'],build)
 save('upstream-build-files.json',filesmap(build));save('private-install-before.json',filesmap(prefix))
 loader=(prefix/'lib/libOpenCL.so.1').resolve();assert loader.is_file();save('selected-loader.json',{'path':str(loader),'sha256':digest(loader)})
 fixture=root/'tests/test_opencl_identity_real.c';old=json.load(open('/tmp/nanolang-opencl-qualified/sources-before.json'))
 for rel in ['tests/test_opencl_identity_real.c','modules/gpu/opencl_runtime.c']:assert digest(root/rel)==old[rel]
 for name,compiler,san in [('gcc-ordinary',cc,False),('gcc-sanitizer',cc,True),('clang-ordinary',clang,False),('clang-sanitizer',clang,True)]:
  binary=out/name;args=[compiler,'-std=c11','-O1','-g','-Wall','-Wextra','-Werror']
  if san:args+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
  args+=['tests/test_opencl_identity_real.c','-ldl','-o',str(binary)]
  phase(name+'-build',args,root,120)
  save(name+'-binary.json',{'path':str(binary),'sha256':digest(binary)})
  phase(name+'-run',[str(binary)],root,120,{'LD_LIBRARY_PATH':str(prefix/'lib')})
  log=(out/(name+'-run.log')).read_text();observed=next(x.split('=',1)[1] for x in log.splitlines() if x.startswith('LIBRARY query='));assert pathlib.Path(observed).resolve()==loader
  assert 'DEVICE name=NVIDIA GB10' in log and 'DEVICE driver=595.84' in log and '48 buffers, 24 kernels, 384 exact integer observations' in log
 status['success']=True
except Exception as error:status['error']=repr(error)
finally:
 after=sources();ta=inventory(toolpaths);la=inventory(libpaths)
 save('source-after.json',after);save('tools-after.json',ta);save('system-libraries-after.json',la)
 if prefix.exists():save('private-install-after.json',filesmap(prefix))
 save('upstream-final-files.json',filesmap(build))
 status.update(sources_unchanged=before==after,tools_unchanged=tb==ta,system_libraries_unchanged=lb==la,head_unchanged=check(['git','rev-parse','HEAD'],root)==head,rows=rows)
 save('status.json',status);print(json.dumps(status),flush=True)
