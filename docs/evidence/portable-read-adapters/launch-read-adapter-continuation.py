import pathlib,subprocess,os,json,time,signal,hashlib,tarfile,shutil
base=pathlib.Path('/private/tmp');old=base/'nanolang-portable-read-adapters-b9ba';new=base/'nanolang-portable-read-adapters-7b66';summary=base/'nanolang-read-adapters-continuation';summary.mkdir(exist_ok=False)
fixture=base/'portable-read-adapters-7b66.c';assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='cc2db12f085dd1045d1bb7ee6f20230690ea206473b4e3b0a8038526369a9660'
env=dict(os.environ,SDKROOT=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip());python='/opt/homebrew/bin/python3'
def run(name,args,cwd):
 state={'argv':args,'cwd':str(cwd),'timeout':False,'errors':[],'signals':[]};start=time.monotonic();p=None
 with (summary/(name+'.log')).open('wb') as f:
  try:
   p=subprocess.Popen(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:p.wait(timeout=600)
   except subprocess.TimeoutExpired:state['timeout']=True
  except Exception as err:state['errors'].append(repr(err))
  finally:
   def exists():
    if p is None:return False
    try:os.killpg(p.pid,0);return True
    except ProcessLookupError:return False
    except OSError as err:state['errors'].append(repr(err));return True
   for sig in [signal.SIGTERM,signal.SIGKILL]:
    if not exists():break
    try:os.killpg(p.pid,sig);state['signals'].append(sig.name)
    except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
     p.poll()
     if not exists():break
     time.sleep(.05)
   state.update(returncode=p.poll() if p else None,group_disappeared=not exists(),seconds=round(time.monotonic()-start,3));(summary/(name+'-terminal.json')).write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps(state),flush=True);assert state['returncode']==0 and not state['timeout'] and not state['errors'] and not state['signals'] and state['group_disappeared']
run('unhooked',[python,str(base/'resume-read-adapter-unhooked.py'),str(old),str(base/'nanolang-read-adapters-b9ba-puck/clang-ordinary'),str(base/'nanolang-read-adapters-7b66-unhooked'),str(fixture)],old)
new.mkdir(exist_ok=False);arc=base/'nanolang-read-adapters-7b66-source.tar.gz'
with tarfile.open(arc) as t:t.extractall(new,filter='data')
shutil.copyfile(base/'nanolang-read-adapters-7b66-tracked.txt',new/'.qualification-tracked')
(summary/'source-extraction.json').write_text(json.dumps({'archive_sha256':hashlib.sha256(arc.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'root':str(new)})+'\n')
run('sanitizer',[python,str(base/'run-read-adapter-sanitizer-only.py'),str(new),str(base/'nanolang-read-adapters-7b66-sanitizer')],new)
