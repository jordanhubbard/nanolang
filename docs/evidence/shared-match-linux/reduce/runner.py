import pathlib,subprocess,hashlib,json,shlex,os,signal,time,sys
root=pathlib.Path('/home/jkh/Src/nanolang-shared-match-reduce-fixture');old=pathlib.Path('/home/jkh/Src/nanolang-shared-match-ordering-controls');prior=pathlib.Path('/tmp/nanolang-match-855-linux-first');out=pathlib.Path('/tmp/nanolang-match-855-linux-reduce');out.mkdir(exist_ok=False)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
def source():return {n:sha(root/n) for n in subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0') if n and (root/n).is_file()}
providers=json.loads((prior/'providers-before.json').read_text());expected_tools=json.loads((prior/'tools-before.json').read_text())
def inputs():return {n:sha(old/n) for n in providers}
def tools():return {p:{'path':str(pathlib.Path(p).resolve()),'sha256':sha(p)} for p in expected_tools}
assert inputs()==providers;assert tools()==expected_tools
pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert pin.startswith('12b61c4ee')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root)
original=source();save('source-before.json',original);save('providers-before.json',providers);save('tools-before.json',expected_tools)
save('reuse.json',{'production_pin':'d0fbd9b1bda184d54563599f2c6f23b41e6ed66d','providers_from':str(old),'original_gate':str(prior),'fixture_pin':pin})
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
def run(name,args,limit):
 start=time.monotonic()
 with (out/(name+'.log')).open('wb') as f:
  p=subprocess.Popen(args,cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code=124
 save(name+'.json',{'argv':args,'exit_code':code,'seconds':time.monotonic()-start});print(name,code,flush=True)
 if code:raise RuntimeError(name+' failed')
status={'pin':pin,'success':False}
try:
 lines=(prior/'build.log').read_text().splitlines();line=next(l for l in reversed(lines) if str(prior/'test_eval') in l and 'tests/test_eval.c' in l)
 args=shlex.split(line)
 for i,a in enumerate(args):
  if a==str(prior/'test_eval'):args[i]=str(out/'test_eval')
  elif a.startswith('obj/') and a.endswith(('.o','.so')):args[i]=str(old/a)
 run('fixture-build',args,120);save('eval-binary.json',{'path':str(out/'test_eval'),'sha256':sha(out/'test_eval')})
 run('evaluator',[str(out/'test_eval')],120)
 code='from pathlib import Path; import unittest; import tests.test_cseed_match_totality as t; t.COMPILER=Path('+repr(str(old/'bin/nanoc_c'))+'); unittest.main(module=t,verbosity=2)'
 run('totality',['/usr/bin/python3','-c',code],600);status['success']=True
except Exception as e:status['error']=str(e)
finally:
 current=source();save('source-after.json',current);status['sources_unchanged']=original==current
 now=inputs();save('providers-after.json',now);status['providers_unchanged']=providers==now
 now=tools();save('tools-after.json',now);status['tools_unchanged']=expected_tools==now
 status['head_unchanged']=pin==subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 save('status.json',status);print(json.dumps(status),flush=True)
sys.exit(0 if status['success'] else 1)
