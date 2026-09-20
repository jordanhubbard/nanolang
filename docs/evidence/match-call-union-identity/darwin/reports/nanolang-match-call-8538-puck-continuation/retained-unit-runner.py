import hashlib,json,os,pathlib,shutil,signal,subprocess,sys,tempfile,time,unittest
out=pathlib.Path(os.environ['MATCH_CALL_REPORT']);out.mkdir(exist_ok=True)
store=out/'artifacts';store.mkdir(exist_ok=True)
work=out/'temporary';work.mkdir(exist_ok=True)
serial=0
OriginalDirectory=tempfile.TemporaryDirectory
class RetainedDirectory(OriginalDirectory):
 def __init__(self,*args,**kwargs):
  kwargs['dir']=work
  super().__init__(*args,**kwargs)
  self._finalizer.detach()
 def cleanup(self):pass
tempfile.TemporaryDirectory=RetainedDirectory
original_run=subprocess.run
def snapshot():
 rows={}
 for p in sorted(work.rglob('*')):
  if p.is_file():
   data=p.read_bytes();digest=hashlib.sha256(data).hexdigest();dest=store/digest
   if not dest.exists():dest.write_bytes(data)
   rows[str(p)]={'sha256':digest,'bytes':len(data),'archive':str(dest)}
 return rows
def run(args,*pargs,**kwargs):
 global serial
 serial+=1;stem=out/f'command-{serial:04d}'
 stem.with_suffix('.before.json').write_text(json.dumps(snapshot(),indent=2))
 requested_check=kwargs.pop('check',False);timeout=kwargs.get('timeout');started=time.monotonic()
 try:
  bound=kwargs.pop('timeout',120)
  if kwargs.pop('capture_output',False):
   kwargs['stdout']=subprocess.PIPE;kwargs['stderr']=subprocess.PIPE
  input_data=kwargs.pop('input',None)
  if input_data is not None:kwargs['stdin']=subprocess.PIPE
  cleanup=[];timed_out=False
  process=subprocess.Popen(args,*pargs,start_new_session=True,**kwargs)
  try:stdout,stderr=process.communicate(input=input_data,timeout=bound)
  except subprocess.TimeoutExpired:
   timed_out=True
   for sig in (signal.SIGTERM,signal.SIGKILL):
    try:os.killpg(process.pid,sig);cleanup.append({'signal':sig.name,'sent':True})
    except ProcessLookupError:cleanup.append({'signal':sig.name,'absent':True})
    try:
     stdout,stderr=process.communicate(timeout=5)
     break
    except subprocess.TimeoutExpired:cleanup.append({'signal':sig.name,'wait_expired':True})
   else:stdout=stderr=None
  result=subprocess.CompletedProcess(args,124 if timed_out else process.returncode,stdout,stderr)
  stem.with_suffix('.cleanup.json').write_text(json.dumps({'timed_out':timed_out,'cleanup':cleanup,'leader_reaped':process.poll() is not None},indent=2))
 except BaseException as error:
  stem.with_suffix('.status.json').write_text(json.dumps({'args':list(map(str,args)),'error':repr(error),'seconds':time.monotonic()-started},indent=2));raise
 finally:
  stem.with_suffix('.after.json').write_text(json.dumps(snapshot(),indent=2))
 for name,data in [('stdout',result.stdout),('stderr',result.stderr)]:
  if data is not None:stem.with_suffix('.'+name).write_bytes(data.encode() if isinstance(data,str) else data)
 stem.with_suffix('.status.json').write_text(json.dumps({'args':list(map(str,args)),'returncode':result.returncode,'seconds':time.monotonic()-started,'timeout':timeout},indent=2))
 if requested_check:result.check_returncode()
 return result
subprocess.run=run
unittest.main(module=None,argv=[sys.argv[0],'-v',*sys.argv[1:]])
