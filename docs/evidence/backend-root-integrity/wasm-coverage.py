from pathlib import Path
import hashlib,json,re,shlex,time
start=time.monotonic();base=Path('/home/jkh/nanolang-qualification/backend-integration-a1bb-combined-seal');reports=base/'reports';index=json.loads((base/'report-sha256.json').read_text());results=[]
configs=[('linux', 'linux/reports/wasm-artifacts.json'), ('darwin', 'puck/reports/wasm-artifacts.json')]
def wasm_no_imports(data):
 assert data[:8]==b'\0asm\x01\0\0\0';pos=8
 def leb(at):
  out=shift=0
  while True:
   assert at<len(data) and shift<35
   v=data[at];at+=1;out|=(v&127)<<shift
   if v<128:return out,at
   shift+=7
 while pos<len(data):
  kind=data[pos];length,begin=leb(pos+1);end=begin+length;assert end<=len(data)
  if kind==2:
   count,after=leb(begin);assert count==0 and after==end
  pos=end
 assert pos==len(data)
for label,path in configs:
 raw=(reports/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==index[path]['sha256'];mapping=json.loads(raw);names={}
 for name,row in mapping.items():names.setdefault(Path(name).name,[]).append((name,row))
 def get(name):
  entries=names[name];assert len(entries)==1,(label,name);key,row=entries[0];data=(base/'objects'/row['sha256']).read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256'];return key,data
 statuses=invocations=workers=recoveries=coverage=products=limited=0
 def status(name):
  global statuses
  _,raw=get(name+'-status.json');d=json.loads(raw);assert d['returncode']==0 and not d['timeout'] and d['leader_reaped'] and d['group_disappeared'] and not d['errors'] and d['launched'],(label,name,d);statuses+=1
 def invoke(name,engine,product,export,args=()):
  global invocations
  status(name);_,raw=get(name+'-command.txt');command=shlex.split(raw.decode());args=list(map(str,args))
  expected=['run','--invoke',export,product,*args] if engine=='wasmtime' else [product,export,*args]
  assert command[-len(expected):]==expected,(label,name,command,expected)
  if engine=='node':assert Path(command[-len(expected)-1]).name=='wasm-run.cjs'
  _,raw=get(name+'-stdout.log');lines=raw.decode().strip().splitlines();assert len(lines)==1;invocations+=1;return int(lines[0])
 for opt in ('O0','O2'):
  for mode in ('linked','observed'):
   prefix='wasm-'+opt+'-'+mode;status(prefix+'-runtime-build')
   expected={f'{prefix}-{i:04d}.wasm' for i in range(73)};actual={n for n in names if re.fullmatch(prefix+r'-\d{4}\.wasm',n)};assert actual==expected
   for i in range(73):
    name=f'{prefix}-{i:04d}';product,data=get(name+'.wasm');wasm_no_imports(data);products+=1
    for stage in ('verify','opt','object','link'):status(name+'-'+stage)
    for engine in ('wasmtime','node'):
     tag=name+'-'+engine;assert invoke(tag+'-run',engine,product,'nano_main')==0
     if mode!='observed':continue
     packed=invoke(tag+'-baseline',engine,product,'nano_baseline_report');calls,peak=packed>>32,packed&0xffffffff;assert calls>0 and peak>8*1024*1024
     memory=invoke(tag+'-memory',engine,product,'nano_memory_report');before,after=memory>>32,memory&0xffffffff;assert before==16 and before<after<=1024
     _,raw=get(tag+'-coverage.json');plan=json.loads(raw);assert plan['complete'] and plan['calls']==calls and plan['requested_live_peak']==peak and plan['engine']==engine and plan['modes']==2 and plan['linear_memory_before_pages']==before and plan['linear_memory_after_pages']==after and plan['page_bytes']==65536
     cursor=total=0
     for begin,end,modes,count in plan['workers']:
      assert cursor==begin and begin<end<=calls and modes==2 and count==2*(end-begin)
      worker=f'{tag}-fault-{begin:06d}-{end:06d}';assert invoke(worker,engine,product,'nano_range_report',(begin,end,calls,peak))==(calls<<32)|count
      cursor=end;total+=count;workers+=1
     assert cursor==calls and total==plan['recoveries']==2*calls;recoveries+=total;coverage+=1
    if mode=='observed' and i==0:
     product,data=get(name+'-limited.wasm');wasm_no_imports(data);status(name+'-limited-link');limited+=1
     for engine in ('wasmtime','node'):assert invoke(name+'-'+engine+'-memory-limit',engine,product,'nano_memory_refusal')==0
 results.append({'host':label,'artifact_report':path,'products':products,'limited_products':limited,'statuses':statuses,'engine_invocations':invocations,'complete_fault_coverages':coverage,'fault_workers':workers,'recoveries':recoveries});print(label,results[-1],flush=True)
out={'status':'PASS','qualified_source_pin':'a1bb1cb85e5551f9e5bb10491f1608877ef6b7b5','hosts':results,'seconds':time.monotonic()-start,'scope':'Actual retained Wasm modules have no imports; both engines execute all73 linked/observed products at O0/O2 with exact command products, successful terminals, dynamic memory growth, contiguous two-mode fault recovery and explicit limited-memory refusal. Startup/ABI/emission/package, source correspondence and current-main integration remain separate obligations.'}
Path('/home/jkh/nanolang-qualification/backend-root-wasm-coverage.json').write_text(json.dumps(out,indent=2)+'\n')
