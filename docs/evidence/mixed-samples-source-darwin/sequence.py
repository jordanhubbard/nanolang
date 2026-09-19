import pathlib,json,subprocess,time,os
root=pathlib.Path('/private/tmp/nanolang-mixed-source-70c511dad'); report=pathlib.Path('/private/tmp/nanolang-mixed-source-darwin-sequence-status.json'); boot=pathlib.Path('/private/tmp/nanolang-mixed-source-darwin-bootstrap-70c511dad/status.json');start=time.monotonic()
while time.monotonic()-start<3600:
 if boot.exists():
  data=json.loads(boot.read_text())
  if 'clean' in data:
   assert data['clean'] and data['sources_unchanged'] and all(x['status']==0 for x in data['steps']),data
   break
 time.sleep(5)
else:raise SystemExit('bootstrap completion deadline')
env=os.environ.copy();env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';results=[]
for name in ('qualify','runtime'):
 print('START',name,flush=True);t=time.monotonic()
 try:r=subprocess.run(['/opt/homebrew/bin/python3','/private/tmp/nanolang-mixed-source-darwin-'+name+'.py'],cwd=root,env=env,timeout=4000);code=r.returncode
 except subprocess.TimeoutExpired:code=124
 results.append({'phase':name,'status':code,'seconds':round(time.monotonic()-t,3)});report.write_text(json.dumps(results,indent=2)+'\n');print('END',name,code,flush=True)
 if code:raise SystemExit(code)
