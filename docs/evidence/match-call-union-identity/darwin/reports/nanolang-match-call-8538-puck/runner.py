import hashlib,json,os,pathlib,platform,shutil,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=False)
darwin=platform.system()=='Darwin'
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
pin=git('rev-parse','HEAD');assert pin.startswith('8538faadf');assert not git('status','--porcelain','--untracked-files=no')
assert not (root/'bin').exists() and not (root/'obj').exists()
tracked=[p for p in git('ls-files').splitlines() if not p.startswith('docs/evidence/')]
def source_map():return {p:sha(root/p) for p in tracked if (root/p).is_file()}
cc='/usr/bin/clang' if darwin else '/bin/gcc'; san='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/bin/gcc'
executables=[shutil.which('cc'),cc,san,sys.executable,shutil.which('python3'),shutil.which('make'),shutil.which('ar'),shutil.which('ld')]
def tool_map():
 paths=set(pathlib.Path(p).resolve() for p in executables if p)
 for folder in ('bin','obj'):
  paths.update(p for p in (root/folder).rglob('*') if p.is_file() and 'nano_modules' not in p.parts and (folder=='bin' or p.suffix in ('.o','.a','.so','.dylib')))
 return {str(p):sha(p) for p in sorted(paths)}
def archive(mapping,name):
 store=out/'artifacts';store.mkdir(exist_ok=True);rows={}
 for path,digest in mapping.items():
  p=pathlib.Path(path);p=p if p.is_absolute() else root/p
  dest=store/digest
  if not dest.exists():shutil.copyfile(p,dest)
  rows[path]={'sha256':digest,'archive':str(dest),'bytes':p.stat().st_size}
 save(name,rows)
env=os.environ.copy()
for k in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH','NANO_BUILD_CACHE','CFLAGS','LDFLAGS'):env.pop(k,None)
env.update(CC=cc,NANOLANG_GUARD_SAN_CC=san,NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'))
env['MATCH_CALL_REPORT']=str(out/'totality-products')
(out/'retained-unit-runner.py').write_bytes(pathlib.Path('/tmp/run-match-call-unit-retained.py').read_bytes())
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes()); before=source_map();save('sources-before.json',before)
archive(before,'source-artifacts.json');save('environment.json',{k:env.get(k) for k in ('CC','NANOLANG_GUARD_SAN_CC','PATH','SDKROOT','NANO_BUILD_CACHE','NANO_MODULE_PATH')})
report={'head':pin,'host':platform.node(),'platform':platform.platform(),'phases':[]};save('manifest.json',report)
for label,args in [('cc',[cc,'--version']),('san',[san,'--version']),('make',['make','--version'])]:
 (out/(label+'-version.txt')).write_bytes(subprocess.check_output(args,stderr=subprocess.STDOUT))
if darwin:
 for label,args in [('sdk-path',['xcrun','--show-sdk-path']),('sdk-version',['xcrun','--show-sdk-version']),('san-runtime',[san,'--print-runtime-dir'])]:
  (out/(label+'.txt')).write_bytes(subprocess.check_output(args))
flags=[] if darwin else ['NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13']
steps=[('bootstrap',['make','-j2','build','CC='+cc,*flags],1800),('userguide-compile',['make','build/userguide/userguide_snippets_check','CC='+cc,*flags],600),('prepare',['make','-j2','nano_virt','nano_vm','nvm2c','CC='+cc,*flags],900),('totality',[sys.executable,'/tmp/run-match-call-unit-retained.py','tests.test_cseed_match_totality','tests.test_shared_match_policy_docs'],1200),('guards',[sys.executable,'-m','unittest','-v','tests.test_canonical_match_guards'],2400),('userguide',['make','userguide-check','CC='+cc,*flags],1800),('full-test',['make','test','CC='+cc,*flags],5400)]
status=1
try:
 for name,command,bound in steps:
  print('START',name,flush=True);t=time.monotonic();tb=tool_map();save(name+'-tools-before.json',tb);save(name+'-sources-before.json',source_map()); cleanup=[];timeout=False
  with (out/(name+'.log')).open('wb') as log:
   phase_env=dict(env)
   if name=='totality':phase_env['PYTHONPATH']=str(root)
   p=subprocess.Popen(command,cwd=root,env=phase_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:p.wait(timeout=bound)
   except subprocess.TimeoutExpired:
    timeout=True
    for sig in (signal.SIGTERM,signal.SIGKILL):
     try:os.killpg(p.pid,sig);cleanup.append({'signal':sig.name,'sent':True})
     except ProcessLookupError:cleanup.append({'signal':sig.name,'absent':True})
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:cleanup.append({'wait_expired':True})
  status=124 if timeout else p.returncode;ta=tool_map();sa=source_map();save(name+'-tools-after.json',ta);save(name+'-sources-after.json',sa);archive(ta,name+'-tool-artifacts.json')
  row={'name':name,'command':command,'status':status,'seconds':round(time.monotonic()-t,3),'timeout':timeout,'cleanup':cleanup,'source_unchanged':sa==before,'tools_unchanged':ta==tb,'log_sha256':sha(out/(name+'.log'))};report['phases'].append(row);save('manifest.json',report);print('END',name,status,row['seconds'],flush=True)
  if status or sa!=before or (name not in ('bootstrap','prepare','userguide-compile','userguide','full-test') and ta!=tb):break
finally:
 save('sources-after.json',source_map());report.update(source_unchanged=source_map()==before,head_unchanged=git('rev-parse','HEAD')==pin,tracked_clean=not git('status','--porcelain','--untracked-files=no'));save('manifest.json',report)
raise SystemExit(status)
