from pathlib import Path
import subprocess,os,json,hashlib,time,re
root=Path.cwd();out=root/'docs/evidence/shadow-stdio-measurement-20261109';work=root/'.test_output/shadow-stdio-measurement';work.mkdir(exist_ok=True)
provider=root/'bin/nanoc_c';library=root/'.test_output/shadow_stdio_probe.dylib'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not os.environ.get('DYLD_INSERT_LIBRARIES'),'I require a baseline without existing loader interposition.'
plan=[{'count':count,'order':['plain','instrumented','instrumented','plain','plain','instrumented']} for count in [1283,5000]]
(out/'plan.json').write_text(json.dumps({'scope':'I measure real descriptor activity in the pinned existing C provider. I do not infer its current-source provenance, whole-compiler performance or release acceptance.','provider':str(provider),'provider_sha256':sha(provider),'library':str(library),'library_sha256':sha(library),'shadow_timeout_seconds':10,'plan':plan},indent=2)+'\n')
records=[]
for group in plan:
 count=group['count'];fixture=work/f'fixture-{count}.nano'
 fixture.write_text('let mut calls:int=0\nfn mark()->bool{set calls (+ calls 1) (println "BODY_SENTINEL") return true}\n'+('shadow mark{assert (mark)}\n'*count)+'fn main()->int{(println calls) return 0}\nshadow main{assert (== calls '+str(count)+')}\n')
 for i,mode in enumerate(group['order']):
  sample=work/f'count-{count}-sample-{i+1}';sample.mkdir(exist_ok=True);product=sample/'program';report=sample/'shadow.json'
  env={**os.environ,'NANO_SHADOW_TIMEOUT_SECONDS':'10'}
  if mode=='instrumented':env['DYLD_INSERT_LIBRARIES']=str(library)
  argv=[str(provider),str(fixture),'--keep-c','--llm-shadow-json',str(report),'-o',str(product)]
  start=time.monotonic();p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120,env=env);elapsed=time.monotonic()-start
  for suffix,data in [('stdout',p.stdout),('stderr',p.stderr)]: (sample/('compile.'+suffix)).write_bytes(data)
  record={'count':count,'sample':i+1,'mode':mode,'argv':argv,'returncode':p.returncode,'wall_seconds':elapsed,'provider_sha256':sha(provider),'fixture_sha256':sha(fixture),'library_sha256':sha(library)}
  records.append(record)
  (out/'samples.json').write_text(json.dumps(records,indent=2)+'\n')
  print(count,i+1,mode,p.returncode,round(elapsed,6),flush=True)
  assert p.returncode==0,p.stdout.decode()+p.stderr.decode()
  summary=json.loads(report.read_text());assert summary['success'] and summary['completed'] and summary['test_count']==count+1
  assert b'BODY_SENTINEL' not in p.stdout+p.stderr
  measurements=[json.loads(m.group()) for m in re.finditer(rb'\{"tool":"nanolang-shadow-stdio-probe"[^\n]*\}',p.stderr)]
  if mode=='instrumented':
   children=[r for r in measurements if r['child'] and r['point']=='_exit' and r['dup']==count+1]
   assert len(children)==1,measurements
   measured=children[0];assert measured['status']==0 and measured['valid_clock']==1
   assert measured['open_devnull']==count+1 and measured['dup2']==2*(count+1)+1 and measured['close']>=2*(count+1)
   record['descriptor_activity']=measured
  else:assert not measurements
  data=product.read_bytes();assert data[:4] in [bytes.fromhex('cffaedfe'),bytes.fromhex('feedfacf')]
  record['product_sha256']=sha(product);run=subprocess.run([str(product)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
  for suffix,data in [('stdout',run.stdout),('stderr',run.stderr)]: (sample/('run.'+suffix)).write_bytes(data)
  assert run.returncode==0 and run.stdout==b'0\n' and sha(product)==record['product_sha256']
  record['run_returncode']=run.returncode;record['shadow_report']=summary
  (out/'samples.json').write_text(json.dumps(records,indent=2)+'\n')
print('I completed every predeclared sample.',flush=True)
