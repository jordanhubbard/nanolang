import pathlib,hashlib,json,os,sys,subprocess,time,signal,shutil
prior=pathlib.Path(sys.argv[1]).resolve();base=pathlib.Path(sys.argv[2]).resolve();base.mkdir(exist_ok=False)
root=prior/'sanitizer/source';out=base
identity=json.loads((prior/'identity.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert identity['pin']=='ead442cf9a53a406385e6f4e0c188b67c5abbdd5'
sources=json.loads((prior/'sanitizer/source-before.json').read_text())
providers=json.loads((prior/'sanitizer/linked-providers-before.json').read_text())
tools=identity['tools']
def verify():
 assert all(digest(root/n)==sha for n,sha in sources.items())
 assert all(digest(root/n)==sha for n,sha in providers.items())
 assert all(digest(pathlib.Path(n).resolve())==sha for n,sha in tools.items())
verify()
assert any(pathlib.Path(n).resolve()==pathlib.Path(sys.executable).resolve() for n in tools)
env=dict(os.environ)
for k in ('CFLAGS','CPPFLAGS','LDFLAGS','SDKROOT','LD_PRELOAD','DYLD_INSERT_LIBRARIES','NANO_SHADOW_TIMEOUT_SECONDS','LSAN_OPTIONS'):
 env.pop(k,None)
env.update(json.loads((prior/'sanitizer/environment.json').read_text()))
compiler=json.loads((prior/'sanitizer/compiler-wrapper.json').read_text())
assert digest(pathlib.Path(compiler['path']))==compiler['sha256']
save(base/'identity.json',dict(source_identity=identity,provider_manifest_sha=digest(prior/'sanitizer/linked-providers-before.json'),source_manifest_sha=digest(prior/'sanitizer/source-before.json'),driver_sha=digest(pathlib.Path(__file__)),scope='Linux remaining original recursive tuple, additive mixed names, original parameter3 only; full TCO sanitizer deadline remains open'))
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
try:
 run('focused-tuple-parameters',[sys.executable,'-m','unittest','-f','-v','tests.test_native_tco.NativeTCO.test_record_and_tuple_parameters','tests.test_native_tco.NativeTCO.test_reused_record_tuple_scalar_callback_parameter_names','tests.test_parameter_nominal_metadata'],root,out,1200)
 save(base/'terminal.json',dict(status=0))
except BaseException as e:
 save(base/'terminal.json',dict(status=1,error=repr(e)));raise
finally:
 verify();assert digest(pathlib.Path(compiler['path']))==compiler['sha256'];save(base/'terminal-invariance.json',dict(sources=True,providers=True,tools=True,wrapper=True))
