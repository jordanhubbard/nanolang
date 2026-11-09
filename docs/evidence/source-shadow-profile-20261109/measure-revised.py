from pathlib import Path
import subprocess,time,os,signal,json,hashlib
root=Path.cwd();out=root/'docs/evidence/source-shadow-profile-20261109';plan=json.loads((out/'revised-plan.json').read_text());work=Path(plan['working_directory']);work.mkdir(parents=True,exist_ok=True);provider=Path(plan['provider'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files=sorted((root/'src_nano').rglob('*.nano'))+[provider,root/'modules/compiler_support/compiler_support.nano',root/'modules/nanoisa/nanoisa.nano'];before={str(p.relative_to(root)):sha(p) for p in files}
env=os.environ.copy();env['NANO_SHADOW_TRACE']='1';env['CC']='/usr/bin/clang';env['NANO_CC']='/usr/bin/clang';env['NANO_MODULE_PATH']=str(root/'modules');env.pop('NANO_SHADOW_TIMEOUT_SECONDS',None)
argv=[plan['argv0_for_existing_root_resolution'],str(root/'src_nano/nanoc_v06.nano'),'--keep-c','-o',str(work/'compiler')];start=time.monotonic();sample=None
record={'executable':str(provider),'argv':argv,'cwd':str(work),'env_overrides':{k:env[k] for k in ['CC','NANO_CC','NANO_MODULE_PATH','NANO_SHADOW_TRACE']},'before':before,'sample_attempted':False,'scope':plan['scope']}
with (out/'revised.stdout').open('wb') as stdout,(out/'revised.stderr').open('wb') as stderr,(out/'revised-sample.stdout').open('wb') as sout,(out/'revised-sample.stderr').open('wb') as serr:
 p=subprocess.Popen(argv,executable=provider,cwd=work,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 while p.poll() is None and time.monotonic()-start<60:
  text=(out/'revised.stderr').read_text(errors='replace')
  if not record['sample_attempted'] and '"event":"shadow-start"' in text.replace(' ',''):
   ids=subprocess.run(['pgrep','-P',str(p.pid)],capture_output=True,text=True).stdout.split()
   if len(ids)==1:
    args=['/usr/bin/sample',ids[0],'8','10','-file',str(out/'revised-child.sample.txt')];sample=subprocess.Popen(args,stdout=sout,stderr=serr);record.update(sample_attempted=True,sample_argv=args,child_pid=int(ids[0]))
  if '"event":"shadow-complete","name":"main","source":"'+str(root/'src_nano/nanoc_v06.nano')+'"' in text.replace(' ',''):
   record['intentional_stop']='I stop only this diagnostic process group after observing the final main shadow completion, before native compilation.';os.killpg(p.pid,signal.SIGTERM);break
  time.sleep(.01)
 if p.poll() is None:
  if 'intentional_stop' not in record:record['diagnostic_timeout']=True;os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=3)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 if sample is not None:
  try:record['sample_returncode']=sample.wait(timeout=20)
  except subprocess.TimeoutExpired:sample.terminate();record['sample_returncode']=sample.wait(timeout=3)
 record.update(returncode=p.wait(),seconds=time.monotonic()-start)
record['after']={str(p.relative_to(root)):sha(p) for p in files};assert record['before']==record['after']
events=[]
for line in (out/'revised.stderr').read_text(errors='replace').splitlines():
 pos=line.find('{')
 if pos<0:continue
 try:j=json.loads(line[pos:])
 except ValueError:continue
 if j.get('event') in ['shadow-start','shadow-complete']:events.append(j)
ends=[e for e in events if e['event']=='shadow-complete'];record.update(starts=sum(e['event']=='shadow-start' for e in events),completes=len(ends),completed_assertion_failures=sum(e['failures'] for e in ends),native_product_exists=(work/'compiler').exists(),summed_recorded_body_seconds=sum(e['elapsed_seconds'] for e in ends));(out/'revised-observation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ['before','after']},indent=2))
