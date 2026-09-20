import os,pathlib,tempfile,subprocess,shutil,json,uuid,hashlib,time
root=pathlib.Path(os.environ['AFFINE_RETAIN_DIR'])
root.mkdir(parents=True,exist_ok=True)
def uid():return str(os.getpid())+'-'+uuid.uuid4().hex
def serial(x):
 if isinstance(x,bytes):return x.decode(errors='replace')
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
 if x is None or isinstance(x,(str,int,float,bool)):return x
 return str(x)
original_temp=tempfile.TemporaryDirectory
class Retained(original_temp):
 def __exit__(self,*args):
  dest=root/('temporary-'+uid())
  shutil.copytree(self.name,dest)
  (root/(dest.name+'.json')).write_text(json.dumps({'original':self.name,'files':{str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}},indent=2)+'\n')
  return super().__exit__(*args)
tempfile.TemporaryDirectory=Retained
original_run=subprocess.run
def retained_run(*args,**kwargs):
 start=time.monotonic(); record={'args':serial(args),'cwd':serial(kwargs.get('cwd'))}
 try:
  result=original_run(*args,**kwargs)
  record.update(status=result.returncode,stdout=serial(result.stdout),stderr=serial(result.stderr));return result
 except BaseException as e:
  record.update(error=repr(e),stdout=serial(getattr(e,'stdout',None)),stderr=serial(getattr(e,'stderr',None)));raise
 finally:
  record['seconds']=time.monotonic()-start
  (root/('command-'+uid()+'.json')).write_text(json.dumps(record,indent=2)+'\n')
subprocess.run=retained_run
