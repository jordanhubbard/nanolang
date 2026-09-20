import hashlib,json,os,pathlib,shutil,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir(exist_ok=False);cas=out/'artifacts';cas.mkdir()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(paths):
 result={}
 for p in paths:
  if not p.is_file():continue
  h=sha(p);dest=cas/h
  if not dest.exists():shutil.copyfile(p,dest)
  result[str(p)]={'sha256':h,'bytes':p.stat().st_size}
 return result
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
assert not (root/'bin').exists() and not (root/'obj').exists()
tracked=[root/p for p in git('ls-files').splitlines() if not p.startswith('docs/evidence/')]
tools=[pathlib.Path(shutil.which(n)).resolve() for n in ('gcc','make','ar','ld','python3')]
source=snapshot(tracked);save('sources-before.json',source);save('tools-before.json',snapshot(tools));shutil.copyfile(__file__,out/'runner.py')
env=os.environ.copy()
for k in list(env):
 if k.startswith(('NANO','NMS_','CC','CFLAGS','LDFLAGS')):env.pop(k,None)
env.update(CC='/bin/gcc',NANO_MODULE_PATH=str(root/'modules'),NANO_BUILD_CACHE=str(root/'obj/module_cache'))
save('environment.json',{k:env.get(k) for k in ('CC','PATH','NANO_MODULE_PATH','NANO_BUILD_CACHE')})
rows=[]
for name,command,bound in [('prepare',['make','-j2','nano_virt','CC=/bin/gcc','CFLAGS=-Wall -Wextra -Werror -std=c99 -g -O3 -ftree-vectorize -fPIC -Isrc -D_GNU_SOURCE -DNANO_SHADOW_PROGRESS_DIAGNOSTIC'],900),('probe',[str(root/'bin/nano_virt'),'tests/diagnostics/emitter_owned_pattern_probe.nano','--root-shadows-only','--emit-nvm','--strip-debug','-o',str(out/'probe.nvm')],120)]:
 before=snapshot([p for f in ('bin','obj') for p in (root/f).rglob('*')]);save(name+'-providers-before.json',before)
 t=time.monotonic();clean=[];timed=False
 with (out/(name+'.log')).open('wb') as log:
  p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:p.wait(timeout=bound)
  except subprocess.TimeoutExpired:
   timed=True
   for sig in (signal.SIGTERM,signal.SIGKILL):
    try:os.killpg(p.pid,sig)
    except ProcessLookupError:pass
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:clean.append(sig.name+' wait expired')
 status=124 if timed else p.returncode
 after=snapshot([p for f in ('bin','obj') for p in (root/f).rglob('*')]);save(name+'-providers-after.json',after)
 rows.append(dict(name=name,command=command,status=status,seconds=round(time.monotonic()-t,3),cleanup=clean,timeout=timed,providers_equal=before==after,output_exists=(out/(name+'.nvm')).exists()))
 save('manifest.json',dict(head=git('rev-parse','HEAD'),phases=rows));print(name,status,flush=True)
 if name=='prepare' and status:break
save('sources-after.json',snapshot(tracked));save('tools-after.json',snapshot(tools));save('outputs.json',snapshot(list(out.glob('*.nvm'))))
assert snapshot(tracked)==source
