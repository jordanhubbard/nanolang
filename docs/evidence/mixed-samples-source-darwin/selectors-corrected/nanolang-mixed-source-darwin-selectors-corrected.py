import pathlib as p,subprocess as s,json,hashlib,os,shlex,time,shutil
root=p.Path('/private/tmp/nanolang-mixed-source-adjacency-70c511dad');out=p.Path('/private/tmp/nanolang-mixed-source-darwin-selectors-corrected-70c511dad');out.mkdir(exist_ok=False)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();git=lambda *a:s.check_output(['git',*a],cwd=root,text=True).strip()
assert git('rev-parse','HEAD')=='70c511dad439f77eba15fe0bdc189e90ff9c9486' and not git('status','--porcelain','--untracked-files=no')
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','Makefile.gnu').splitlines();sources=lambda:{f:sha(root/f) for f in files if (root/f).is_file()};inputs=lambda:{str(f):sha(f) for n in ('bin','obj','lib') for f in (root/n).rglob('*') if f.is_file()}
before=sources();initial=inputs();(out/'sources-before.json').write_text(json.dumps(before,indent=2)+'\n');(out/'inputs-before.json').write_text(json.dumps(initial,indent=2)+'\n');shutil.copyfile(__file__,out/p.Path(__file__).name)
line=next(x for x in p.Path('/private/tmp/nanolang-mixed-source-darwin-adjacency-corrected-70c511dad/service.log').read_text().splitlines() if x.startswith('SERVICE_MODULE_OBJECTS='));tokens=shlex.split(line);prepared={k:v for k,v in (tokens[0].split('=',1),tokens[1].split('=',1))};assert set(prepared)=={'SERVICE_MODULE_OBJECTS','SERVICE_MODULE_LDFLAGS'}
env=os.environ.copy();env.update(prepared);env.update(PYTHONPATH=str(root),CC='/opt/homebrew/opt/llvm/bin/clang',NANO_SERVICE_MODULE_TEST_CC='/opt/homebrew/opt/llvm/bin/clang',NANO_SERVICE_MODULE_TEST_CFLAGS='',NANO_MIXED_RUNTIME_DIR=str(out/'native'),SDKROOT=s.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip());(out/'native').mkdir()
(out/'environment.json').write_text(json.dumps({k:env[k] for k in (*prepared,'CC','NANO_SERVICE_MODULE_TEST_CC','NANO_SERVICE_MODULE_TEST_CFLAGS','NANO_MIXED_RUNTIME_DIR','SDKROOT')},indent=2)+'\n')
tools={str(f):{'resolved':str(f.resolve()),'sha256':sha(f)} for f in [p.Path('/opt/homebrew/opt/llvm/bin/clang'),p.Path('/opt/homebrew/bin/python3')]};(out/'tools-before.json').write_text(json.dumps(tools,indent=2)+'\n')
commands=[('service',['/opt/homebrew/bin/python3','-m','unittest','-f','-v','tests.test_service_bindings_module']),('mixed-native-lsan',['/opt/homebrew/bin/python3','-m','unittest','-f','-v','tests.test_mixed_samples_runtime'])];report={'pin':git('rev-parse','HEAD'),'steps':[]};status=1
try:
 for name,cmd in commands:
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as log:
   try:status=s.run(cmd,cwd=root,env=env,stdout=log,stderr=s.STDOUT,timeout=600).returncode
   except s.TimeoutExpired:status=124
  report['steps'].append({'name':name,'command':cmd,'status':status,'seconds':round(time.monotonic()-start,3)});(out/'status.json').write_text(json.dumps(report,indent=2)+'\n');print('END',name,status,flush=True)
  if status:break
finally:
 after=sources();final=inputs();(out/'sources-after.json').write_text(json.dumps(after,indent=2)+'\n');(out/'inputs-after.json').write_text(json.dumps(final,indent=2)+'\n');(out/'tools-after.json').write_text(json.dumps({f:{'resolved':str(p.Path(f).resolve()),'sha256':sha(p.Path(f))} for f in tools},indent=2)+'\n');report.update(sources_unchanged=before==after,inputs_unchanged=initial==final,clean=not git('status','--porcelain','--untracked-files=no'));(out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(status)
