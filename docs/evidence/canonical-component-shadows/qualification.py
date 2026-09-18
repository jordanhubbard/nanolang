from pathlib import Path
import subprocess,json,hashlib,time,sys,os,signal,atexit
root=Path('/home/jkh/Src/nanolang-canonical-component-shadows')
out=Path('/tmp/nanolang-canonical-component-shadows');out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
report={'head':git('rev-parse','HEAD'),'production':'e9a5f55f9b82b3de8c03b3df240665865fd25e5c','source_clean_before':not git('status','--porcelain'),'steps':[]}
def save(): (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
def finish():
 report['head_unchanged']=git('rev-parse','HEAD')==report['head'];report['source_clean_after']=not git('status','--porcelain')
 if 'tools_before' in report:
  report['tools_after']={f:sha(root/f) for f in report['tools_before']};report['tools_unchanged']=report['tools_after']==report['tools_before']
 save()
atexit.register(finish)
def run(name,args,limit,env=None):
 assert git('rev-parse','HEAD')==report['head'] and not git('status','--porcelain')
 log=out/(name+'.log');start=time.monotonic();print(name,'started',flush=True)
 with log.open('w') as f:
  p=subprocess.Popen([str(a) for a in args],cwd=root,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  try:code=p.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code=124
 step={'name':name,'args':[str(a) for a in args],'limit':limit,'status':code,'seconds':round(time.monotonic()-start,3),'log':str(log),'sha256':sha(log)};report['steps'].append(step);save();print(name,code,flush=True)
 if code:sys.exit(code)
assert report['source_clean_before'];save()
run('bootstrap',['make','-j8','bootstrap'],1800)
run('tools',['make','-j8','nano_vm','nanoisa_dump'],1800)
assert (root/'bin/nanoc').resolve()==root/'bin/nanoc_stage2'
report['tools_before']={f:sha(root/f) for f in ['bin/nanoc_c','bin/nanoc_stage1','bin/nanoc_stage2','bin/nano_vm','bin/nanoisa']};save()
hook=out/'capture-vm.py'
hook.write_text('''#!/usr/bin/env python3
from pathlib import Path
import os,sys,shutil,subprocess,json,hashlib,time
assert len(sys.argv)==3 and sys.argv[1]=='--check-shadows'
p=Path(os.environ['NANO_SHADOW_CAPTURE'])
shutil.copyfile(sys.argv[2],p.with_suffix('.nvm'))
t=time.monotonic()
r=subprocess.run([os.environ['NANO_SHADOW_REAL_VM'],*sys.argv[1:]])
p.with_suffix('.json').write_text(json.dumps({'arguments':sys.argv[1:],'module_sha256':hashlib.sha256(p.with_suffix('.nvm').read_bytes()).hexdigest(),'vm_status':r.returncode,'seconds':round(time.monotonic()-t,3)},indent=2)+'\\n')
sys.exit(r.returncode)
''');hook.chmod(0o755)
report['hook_sha256']=sha(hook);save()
for component in ['parser','typecheck','transpiler']:
 source=root/f'src_nano/{component}_driver.nano';target=out/(component+'.nvm');capture=out/(component+'-shadows')
 report.setdefault('sources',{})[str(source.relative_to(root))]=sha(source);save()
 env=dict(os.environ,NANO_VM=str(hook),NANO_SHADOW_CAPTURE=str(capture),NANO_SHADOW_REAL_VM=str(root/'bin/nano_vm'))
 run(component+'-compile',[root/'bin/nanoc_stage2',source,'--emit-nvm','--test-imports','--verbose','-o',target],1200,env)
 receipt=json.loads(capture.with_suffix('.json').read_text());assert receipt['vm_status']==0
 report.setdefault('shadow_reports',{})[component]=receipt;report.setdefault('outputs',{})[component]=sha(target);save()
 run(component+'-shadow-dump',[root/'bin/nanoisa','dump',capture.with_suffix('.nvm')],60)
 run(component+'-entry',[root/'bin/nano_vm',target],60)
report['complete']=True;save()
