from pathlib import Path
import hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
base=Path(__file__).resolve().parent
qualified=Path('/tmp/nanolang-literal-order-75051f193')
source=qualified/'source'
if len(sys.argv)>1 and sys.argv[1]=='--child':
 spec=importlib.util.spec_from_file_location('literal_optimization',base/'fixture.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 mod.ROOT=source
 import unittest
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(mod.NativeLiteralOptimization)
 raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
reports=base/'reports';reports.mkdir(exist_ok=False)
def write(name,obj): (reports/name).write_text(json.dumps(obj,indent=2)+'\n')
for name in ('bootstrap','native-gate'):
 result=json.loads((qualified/'reports'/(name+'.status.json')).read_text())
 assert result['returncode']==0 and not result.get('timeout'),result
manifest=json.loads((qualified/'manifest.json').read_text())
def snapshot():
 out={}
 for row in manifest['files']:
  path=source/row['path'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
  assert digest==row['sha256'] and path.stat().st_mode&0o777==row['mode'],row['path']
  out[row['path']]=digest
 return out
write('source-before.json',snapshot())
assert hashlib.sha256((base/'fixture.py').read_bytes()).hexdigest()==json.loads((base/'reuse.json').read_text())['fixture_sha256']
env=os.environ.copy();env['PATH']='/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin'
for key in ('NANOLANG_ROOT','NANOLANG_SDK_ROOT','NANO_MODULE_PATH','NANOC','NANO_CFLAGS','NANO_LDFLAGS','NANO_CC','CC'):env.pop(key,None)
free=shutil.disk_usage(base).free;assert free>=2*1024**3,free
row={'command':[sys.executable,str(base/'run.py'),'--child'],'cwd':str(source),'timeout_seconds':600,'free_before':free,'python':sys.executable,'PATH':env['PATH']}
start=time.monotonic()
with (reports/'optimization.log').open('wb') as log:
 p=subprocess.Popen(row['command'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 row['pid']=p.pid;write('status.json',row)
 while p.poll() is None:
  if time.monotonic()-start>600 or shutil.disk_usage(base).free<1024**3:
   row['stop_reason']='timeout' if time.monotonic()-start>600 else 'capacity'
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  time.sleep(1)
 row['returncode']=p.wait();row['seconds']=time.monotonic()-start;row['free_after']=shutil.disk_usage(base).free
write('status.json',row);write('source-after.json',snapshot())
print(json.dumps(row),flush=True)
raise SystemExit(row['returncode'] or (1 if 'stop_reason' in row else 0))
