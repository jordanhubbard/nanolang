import hashlib,json,os,pathlib,platform,shlex,shutil,signal,subprocess,sys,tarfile,time
base=pathlib.Path(sys.argv[1]).resolve(); archive=pathlib.Path(sys.argv[2]).resolve();base.mkdir(exist_ok=False)
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mapping(root):return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and not p.is_symlink()}
darwin=platform.system()=='Darwin';env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
env.pop('LSAN_OPTIONS',None)
for key in ('CFLAGS','CPPFLAGS','LDFLAGS','SDKROOT','LD_PRELOAD','DYLD_INSERT_LIBRARIES','NANOLANG_COMPILER','NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_CFLAGS','NANO_SHADOW_TIMEOUT_SECONDS','NANO_SHADOW_TIMING'):
 env.pop(key,None)
if darwin:env['PATH']='/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
make=shutil.which('make',path=env['PATH']);cc='/usr/bin/clang' if darwin else '/usr/bin/gcc';sancc='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/usr/bin/gcc'
sdk=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip() if darwin else ''
save(base/'identity.json',dict(pin='acc1355dc',archive_sha=digest(archive),driver_sha=digest(pathlib.Path(__file__)),host=platform.uname()._asdict(),sdk=sdk,tools={p:digest(pathlib.Path(p).resolve()) for p in [make,cc,sancc,sys.executable]}))
assert shutil.disk_usage(base).free>2*1024**3
records=[]
def run(label,args,root,out,bound):
 status=dict(label=label,argv=args,bound=bound,timeout=False);start=time.monotonic();seen=set();p=None
 try:
  with (out/(label+'.log')).open('wb') as log:
   p=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   status['pid']=p.pid;save(out/(label+'.json'),status)
   while p.poll() is None:
    rows=subprocess.check_output(['ps','-axo','pid=,ppid='],text=True);pairs=[tuple(map(int,r.split())) for r in rows.splitlines() if len(r.split())==2];parents={p.pid};changed=True
    while changed:
     child={a for a,b in pairs if b in parents};new=child-parents;changed=bool(new);parents|=new
    seen|=parents
    if time.monotonic()-start>bound:status['timeout']=True;break
    if shutil.disk_usage(base).free<1536*1024**2:status['capacity_stop']=True;break
    time.sleep(.2)
 finally:
  if p:
   for sig in (signal.SIGTERM,signal.SIGKILL):
    for pid in seen|{p.pid}:
     try:os.killpg(pid,sig)
     except ProcessLookupError:pass
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
   status['returncode']=p.poll();status['remaining_groups']=[]
   for pid in seen|{p.pid}:
    try:os.killpg(pid,0);status['remaining_groups'].append(pid)
    except ProcessLookupError:pass
  status['seconds']=time.monotonic()-start;save(out/(label+'.json'),status);records.append(status);save(base/'phases.json',records)
 print(label,status,flush=True)
 assert status.get('returncode')==0 and not status['timeout'] and not status.get('capacity_stop') and not status['remaining_groups'],status
 text=(out/(label+'.log')).read_text(errors='replace');assert 'ERROR: AddressSanitizer' not in text and 'runtime error:' not in text and 'ERROR: LeakSanitizer' not in text
linked_before=None
def linked_closure(root):
 return {str(p.relative_to(root)):digest(p) for folder in ('bin','obj','lib')
         for p in (root/folder).rglob('*') if p.is_file() and not p.is_symlink()
         and (folder!='obj' or p.suffix=='.o')}
def verify_linked_terminal():
 if linked_before is None:return
 after={name:digest(root/name) for name in linked_before if (root/name).is_file()}
 missing=sorted(set(linked_before)-set(after))
 changed=sorted(name for name in after if after[name]!=linked_before[name])
 save(out/'linked-providers-after-terminal.json',dict(files=after,missing=missing,changed=changed))
 for fixture in (root/'obj/test_cop_exec',root/'obj/exec_provider.so',root/'obj/test_cop_opaque',root/'obj/opaque_provider.so',root/'obj/test_cop_lifecycle'):
  if fixture.is_file():
   retained=out/'retained';retained.mkdir(exist_ok=True)
   shutil.copy2(fixture,retained/fixture.name)
   save(out/(fixture.name+'-executable.json'),dict(path=str(fixture),sha256=digest(fixture),retained_sha256=digest(retained/fixture.name)))

 if missing or changed:raise RuntimeError('I refuse changed linked provider closure')
try:
 for name,compiler,san in [('ordinary',cc,False),('sanitizer',sancc,True)]:
  linked_before=None
  out=base/name;out.mkdir();root=out/'source';root.mkdir()
  with tarfile.open(archive) as tf:tf.extractall(root)
  sources=mapping(root);save(out/'source-before.json',sources)
  tmp=out/'tmp';tmp.mkdir();env['TMPDIR']=str(tmp);env['NANOLANG_SDK_ROOT']=str(root)
  env['NANO_BUILD_CACHE']=str(root/'obj/module_cache');env['NANO_MODULE_PATH']=str(root/'modules')
  flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-sanitize-recover=all'] if san else []
  sdkflags=['-isysroot',sdk] if darwin else []
  if darwin:env['SDKROOT']=sdk
  cflags=['-Wall','-Wextra','-Werror','-std=c99','-g','-O1','-fPIC','-Isrc','-D_GNU_SOURCE',*sdkflags,*flags]
  args=[make,'-f','Makefile.gnu','-j2','CC='+compiler,'PYTHON_WITH_YAML='+sys.executable,'CFLAGS='+shlex.join(cflags),'LDFLAGS='+shlex.join(['-lm',*sdkflags,*flags])]
  # Actual Make compiler is unwrapped: assembler-capture's existing sanitizer exclusion remains effective.
  for key in ('NANO_CC','NANO_NATIVE_TEST_CC','NANO_LDFLAGS'):env.pop(key,None)
  env['PATH']=('/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin' if darwin else os.environ['PATH'])
  run('fresh-providers',[*args,'bin/nanoc_c','nano_virt','nano_vm','nano_vmd','nano_cop','nvm2c','nanoisa_dump','nvm2c-runtime'],root,out,1800)
  linked_before=linked_closure(root);save(out/'linked-providers-before.json',linked_before)
  wrapper_dir=out/'tools';wrapper_dir.mkdir();wrapper=wrapper_dir/'cc'
  wrapper.write_text('#!/bin/sh\nexec '+shlex.join([compiler,*sdkflags,*flags])+' "$@"\n');wrapper.chmod(0o755)
  env['PATH']=str(wrapper_dir)+':'+env['PATH'];env['NANO_CC']=str(wrapper);env['NANO_NATIVE_TEST_CC']=str(wrapper)
  env['NANO_LDFLAGS']=shlex.join(flags);env['NANO_AS_CAPTURE_HELPER']=str(root/'bin/nano_as_capture.so')
  env['NANO_ARTIFACT_CFLAGS']='';env['NANO_ARTIFACT_LDFLAGS']=''
  env['NANO_SDK_PROBE_CHILD_MODE']='callback-explicit-child' if darwin and san else 'callback'
  save(out/'probe-mode.json',dict(mode=env['NANO_SDK_PROBE_CHILD_MODE']))
  save(out/'environment.json',{k:v for k,v in env.items() if k.startswith(('NANO','ASAN','UBSAN','LSAN')) or k in ('PATH','SDKROOT','TMPDIR')})
  save(out/'compiler-wrapper.json',dict(path=str(wrapper),sha256=digest(wrapper),body=wrapper.read_text()))
  before=mapping(root/'bin');save(out/'providers-before.json',before)
  run('exec-owning',[*args,'test-cop-exec'],root,out,1200)
  exec_log=(out/'exec-owning.log').read_text(errors='replace')
  assert exec_log.count('exit 0 before parent cleanup.')==6,exec_log
  run('original-opaque',[*args,'test-cop-opaque'],root,out,1200)
  run('original-protocol',[*args,'test-cop-protocol'],root,out,1200)
  run('original-fuzz',[*args,'test-cop-fuzz'],root,out,1200)
  run('original-ffi',[*args,'test-vm-ffi'],root,out,1200)
  run('original-lifecycle',[*args,'test-cop-lifecycle'],root,out,1200)
  run('borrowed-string-controls',[sys.executable,'-m','unittest','-f','-v','tests.test_compiler_support_artifact_adapters.CompilerSupportArtifactAdapters.test_borrowed_string_arguments_and_isolated_snapshots'],root,out,1200)
  after={n:digest(root/n) for n in sources};save(out/'source-after.json',after);assert after==sources
  save(out/'providers-after.json',mapping(root/'bin'));assert mapping(root/'bin')==before
  verify_linked_terminal()
 save(base/'terminal.json',dict(status=0,scope='fresh ordinary and fully instrumented exec ownership, actual retained callback coexistence, original opaque/protocol/fuzz/FFI/lifecycle with daemon clients, borrowed strings; installed source-hidden control follows as separate actual full-install prerequisite'))

except BaseException as e:
 verify_linked_terminal()
 if 'root' in globals() and 'sources' in globals():
  save(out/'source-after-terminal.json',{n:digest(root/n) for n in sources})
  save(out/'products-terminal.json',{str(p.relative_to(root)):digest(p) for folder in ['bin','obj'] for p in (root/folder).rglob('*') if p.is_file()})
 save(base/'terminal.json',dict(status=1,error=repr(e)));raise
finally:
 recorded=json.loads((base/'identity.json').read_text())['tools']
 terminal_tools={name:digest(pathlib.Path(name).resolve()) for name in recorded}
 save(base/'tools-after-terminal.json',dict(tools=terminal_tools,unchanged=terminal_tools==recorded))
 if terminal_tools!=recorded:raise RuntimeError('I refuse changed compiler/Make/Python endpoints')
