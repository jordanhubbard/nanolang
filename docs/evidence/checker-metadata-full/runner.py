import pathlib,json,hashlib,subprocess,shlex,os,time,signal
r=pathlib.Path('/home/jkh/Src/nanolang-checker-metadata-ownership');prior=pathlib.Path('/tmp/nanolang-checker-owner-f7c-gates');o=pathlib.Path('/tmp/nanolang-checker-owner-full-metadata');o.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x):(o/n).write_text(json.dumps(x,indent=2)+'\n')
(o/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
providers=json.loads((prior/'adjacent-providers-after.json').read_text());sources=json.loads((prior/'source-after.json').read_text());tools=json.loads((prior/'tools-after.json').read_text())
def verify():
 assert all(sha(r/p)==s for p,s in sources.items())
 assert all(sha(r/p)==s for p,s in providers.items())
 assert all(sha(pathlib.Path(v['resolved']))==v['sha256'] for v in tools.values())
 return {'source':sources,'providers':providers,'tools':tools}
write('before.json',verify())
line=next(s for s in (prior/'gcc-sanitized.log').read_text().splitlines() if ' -o obj-checker-gcc/test_checker_metadata_ownership ' in s)
cmd=shlex.split(line);cmd[0]='/usr/bin/gcc';cmd[cmd.index('-o')+1]=str(o/'test_module_metadata');cmd[cmd.index('tests/test_checker_metadata_ownership.c')]='tests/test_module_metadata.c'
env=os.environ.copy();env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1');env.pop('LSAN_OPTIONS',None)
write('environment.json',{k:env.get(k) for k in ['ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS']})
report={'pin':'f7c58a0de2ecdeca9ce30d25d33c83d6c45372d8','precode':'b53ce04cf','steps':[]};status=1
try:
 for name,args in [('build',cmd),('run',[str(o/'test_module_metadata')])]:
  print('START',name,flush=True);t=time.monotonic()
  with (o/(name+'.log')).open('wb') as log:
   p=subprocess.Popen(args,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:status=p.wait(timeout=300)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    status=124
  report['steps'].append({'name':name,'command':args,'status':status,'seconds':round(time.monotonic()-t,3),'log_sha256':sha(o/(name+'.log'))});write('manifest.json',report);print('END',name,status,flush=True)
  if status:break
finally:
 write('after.json',verify());report['all_inputs_unchanged']=True
 if (o/'test_module_metadata').exists():report['binary_sha256']=sha(o/'test_module_metadata')
 write('manifest.json',report)
raise SystemExit(status)
