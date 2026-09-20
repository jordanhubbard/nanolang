import hashlib,json,os,pathlib,shlex,shutil,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]).resolve(); evidence=pathlib.Path(sys.argv[2]).resolve(); platform=sys.argv[3]; prior=pathlib.Path(sys.argv[4]).resolve(); fixture=pathlib.Path(sys.argv[5]).resolve()
evidence.mkdir(parents=True,exist_ok=False); store=evidence/'objects';store.mkdir()
def dump(name,value): (evidence/name).write_text(json.dumps(value,indent=2)+'\n')
def item(p):
 p=pathlib.Path(p).resolve();data=p.read_bytes();h=hashlib.sha256(data).hexdigest();dest=store/h
 if not dest.exists():dest.write_bytes(data)
 return {'sha256':h,'bytes':len(data),'archive':str(dest)}
def snapshot(paths):return {str(p):item(p) for p in sorted(set(pathlib.Path(p).resolve() for p in paths),key=str)}
manifest=[]
def command(label,argv,env=None,timeout=180):
 state={'label':label,'argv':list(map(str,argv)),'timeout':False,'cleanup_complete':True,'returncode':None};start=time.monotonic()
 dump(label+'-command.json',state)
 try:
  with (evidence/(label+'.log')).open('wb') as output:
   proc=subprocess.Popen(state['argv'],cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
   try:state['returncode']=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    state['timeout']=True
    for sig in (signal.SIGTERM,signal.SIGKILL):
     try:os.killpg(proc.pid,sig)
     except ProcessLookupError:pass
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:continue
    state['returncode']=124;state['child_returncode']=proc.poll();state['cleanup_complete']=proc.poll() is not None
   try:os.killpg(proc.pid,0)
   except ProcessLookupError:pass
   else:
    state['cleanup_complete']=False
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
 except OSError as error:state['error']=repr(error);state['cleanup_complete']=False
 state['seconds']=time.monotonic()-start;dump(label+'-status.json',state);manifest.append(state);dump('manifest.json',manifest)
 print(label,state['returncode'],round(state['seconds'],3),flush=True)
 return state['returncode']==0 and state['cleanup_complete']

basefile='nanoisa-after.json' if platform=='linux' else 'homebrew-sanitizer-after.json'
baseline=json.loads((prior/basefile).read_text()); inputs=list(baseline)
if platform!='linux':
 productmap=json.loads((prior/'homebrew-sanitizer/10-products.json').read_text())
 for name,row in productmap.items():
  if name.startswith('linked-') and name.endswith('.o'):
   p=pathlib.Path(row['path']); inputs.append(str(p))
   if item(p)['sha256']!=row['sha256']:raise RuntimeError('Prior product drift: '+str(p))
inputs.append(str(fixture));before=snapshot(inputs)
for path,row in baseline.items():
 if before[path]['sha256']!=row['sha256']:raise RuntimeError('Prior input drift: '+path)
dump('before.json',before);dump('driver.json',item(__file__));dump('fixture.json',item(fixture))
env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='')
ok=True
try:
 if platform!='linux':
  argv=json.loads((prior/'homebrew-sanitizer/10-status.json').read_text())['argv']
  argv=[str(fixture) if x=='tests/nanoisa/test_portable_host_plan.c' else x for x in argv]
  binary=evidence/'homebrew-linked-binary';argv[argv.index('-o')+1]=str(binary)
  dump('reuse.json',{'prior_command':str(prior/'homebrew-sanitizer/10-status.json'),'changes':'Only corrected fixture path and fresh output binary; original four unhooked objects verified from retained product hashes.'})
  phase_before=snapshot(inputs);dump('homebrew-linked-before.json',phase_before)
  ok=command('homebrew-linked-build',argv,env)
  if ok:ok=command('homebrew-linked-run',[binary],env)
  if binary.exists():dump('homebrew-linked-binary.json',item(binary))
  phase_after=snapshot(inputs);dump('homebrew-linked-after.json',phase_after);ok=ok and phase_before==phase_after
 if ok:
  providers=[root/p for p in shlex.split((prior/'providers.txt').read_text())]
  flags=shlex.split((prior/'cflags.txt').read_text());ldflags=shlex.split((prior/'ldflags.txt').read_text())
  cases=[('nanoisa','tests/nanoisa/test_nanoisa.c',providers)]
  if platform!='linux':cases=[('generic-verifier','tests/nanoisa/test_verifier.c',providers),('verifier-cleanup','tests/nanoisa/test_verifier_cleanup.c',[p for p in providers if p.name!='verifier.o'])]+cases
  for label,source,selected in cases:
   cc='/bin/gcc' if platform=='linux' else '/usr/bin/clang';binary=evidence/(label+'-binary')
   phase_before=snapshot(inputs);dump(label+'-before.json',phase_before)
   ok=command(label+'-build',[cc,*flags,'-Isrc/nanoisa','-Imodules/nanoisa',source,*selected,*ldflags,'-o',binary],env)
   if ok:ok=command(label+'-run',[binary],env)
   if binary.exists():dump(label+'-binary.json',item(binary))
   phase_after=snapshot(inputs);dump(label+'-after.json',phase_after);ok=ok and phase_before==phase_after
   if not ok:break
finally:
 after=snapshot(inputs);dump('after.json',after)
 dump('terminal.json',{'status':'PASS' if ok and before==after else 'FAILED','inputs_equal':before==after,'earlier_passes':'Remain attributed to e9e; new fixture assertion executed only by fresh Homebrew linked binary.'})
if not ok or before!=after:sys.exit(1)
