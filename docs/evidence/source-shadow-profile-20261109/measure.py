from pathlib import Path
import subprocess,time,os,signal,json,hashlib
root=Path.cwd();out=root/'docs/evidence/source-shadow-profile-20261109';work=root/'.test_output/source-shadow-profile';provider=work/'nanoc_c'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=sorted((root/'src_nano').rglob('*.nano'))+[provider,root/'modules/compiler_support/compiler_support.nano',root/'modules/nanoisa/nanoisa.nano']
before={str(p.relative_to(root)):sha(p) for p in files}
env=os.environ.copy();env.update(CC='/usr/bin/false',NANO_SHADOW_TRACE='1');env.pop('NANO_SHADOW_TIMEOUT_SECONDS',None)
argv=[str(provider),str(root/'src_nano/nanoc_v06.nano'),'--keep-c','-o',str(work/'compiler')]
start=time.monotonic();observation={'argv':argv,'scope':'I diagnose actual supervised shadows; CC=false and relocated runtime roots intentionally prevent native creation acceptance.','before':before,'sample_attempted':False}
with (out/'compile.stdout').open('wb') as stdout,(out/'compile.stderr').open('wb') as stderr:
 p=subprocess.Popen(argv,cwd=root,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
 while p.poll() is None and time.monotonic()-start<90:
  text=(out/'compile.stderr').read_text(errors='replace')
  if not observation['sample_attempted'] and '"event":"shadow-start"' in text.replace(' ',''):
   kids=subprocess.run(['pgrep','-P',str(p.pid)],capture_output=True,text=True)
   ids=[int(x) for x in kids.stdout.split()]
   if len(ids)==1:
    child=ids[0];args=['/usr/bin/sample',str(child),'5','10','-file',str(out/'child.sample.txt')]
    sample=subprocess.run(args,capture_output=True,timeout=20)
    (out/'sample.stdout').write_bytes(sample.stdout);(out/'sample.stderr').write_bytes(sample.stderr)
    observation.update(sample_attempted=True,sample_argv=args,sample_returncode=sample.returncode,child_pid=child)
  time.sleep(.05)
 if p.poll() is None:
  observation['diagnostic_timeout']=True
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=3)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 observation.update(returncode=p.wait(),seconds=time.monotonic()-start)
observation['after']={str(p.relative_to(root)):sha(p) for p in files};assert before==observation['after']
events=[]
for line in (out/'compile.stderr').read_text(errors='replace').splitlines():
 pos=line.find('{')
 if pos<0:continue
 try:obj=json.loads(line[pos:])
 except ValueError:continue
 if obj.get('event') in ['shadow-start','shadow-complete']:events.append(obj)
complete=[e for e in events if e['event']=='shadow-complete']
observation.update(starts=sum(e['event']=='shadow-start' for e in events),completes=len(complete),assertion_failures=sum(e['failures'] for e in complete),summed_body_seconds=sum(e['elapsed_seconds'] for e in complete),native_product_exists=(work/'compiler').exists())
(out/'observation.json').write_text(json.dumps(observation,indent=2)+'\n');print(json.dumps({k:v for k,v in observation.items() if k not in ['before','after']},indent=2))
