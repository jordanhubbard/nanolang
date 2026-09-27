from pathlib import Path
import hashlib,json,re,shlex,time
start=time.monotonic();base=Path('/run/user/1000/nanolang-record-llvm-complete-seal');reports=base/'reports';index=json.loads((base/'report-sha256.json').read_text());result=[]
ordinary='nanolang-record-llvm-linux-ordinary-seal/nanolang-record-llvm-78042-linux/'
darwin='nanolang-record-llvm-darwin-recovery-seal/nanolang-record-llvm-78042-recovery-puck-backup/'
san='nanolang-record-llvm-linux-sanitizer-seal/'
configs=[('linux-gcc',ordinary+'ordinary-native-artifacts.json',('O0','O2')),('linux-clang',ordinary+'clang-ordinary-native-artifacts.json',('O0','O2')),('darwin-apple',darwin+'ordinary-native-artifacts.json',('O0','O2')),('darwin-homebrew',darwin+'clang-ordinary-native-artifacts.json',('O0','O2')),('darwin-homebrew-sanitized',darwin+'sanitizer-native-artifacts.json',('O0','O2')),('linux-gcc-sanitized',san+'nanolang-record-llvm-78042-linux-sanitizers/sanitizer-native-artifacts.json',('O0','O2')),('linux-clang-sanitized-O0',san+'nanolang-record-llvm-78042-linux-clang-continuation-r2/clang-sanitizer-native-artifacts.json',('O0',)),('linux-clang-sanitized-O2',san+'reports/clang-sanitizer-native-artifacts.json',('O2',))]
for label,path,opts in configs:
 data=(reports/path).read_bytes();assert hashlib.sha256(data).hexdigest()==index[path]['sha256'];mapping=json.loads(data);byname={}
 for name,row in mapping.items():byname.setdefault(Path(name).name,[]).append((name,row))
 def get(name):
  rows=byname[name];assert len(rows)==1,(label,name);key,row=rows[0];p=base/'objects'/row['sha256'];data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256'];return key,data
 statuses=workers=recoveries=0;coverage=products=0
 def status(name):
  global statuses
  _,data=get(name+'-status.json');d=json.loads(data)
  assert d['returncode']==0 and not d['timeout'] and d['leader_reaped'] and d['group_disappeared'] and not d['errors'] and d['launched'],(label,name,d)
  statuses+=1
 for opt in opts:
  for mode in ('linked','observed'):
   expected={f'{opt}-{mode}-{i:04d}' for i in range(73)}
   actual={n for n in byname if re.fullmatch(opt+'-'+mode+r'-\d{4}',n)};assert actual==expected,(label,opt,mode,len(actual))
   status(opt+'-'+mode+'-runtime-build')
   for i in range(73):
    name=f'{opt}-{mode}-{i:04d}';product,_=get(name);products+=1
    for stage in ('verify','opt','object','link','nm'):status(name+'-'+stage)
    _,nm=get(name+'-nm-stdout.log');assert all(v not in nm for v in (b'vm_core_execute',b'vm_record_array',b'isa_decode',b'nvm_prepare'))
    if mode=='linked':
     status(name+'-run');_,command=get(name+'-run-command.txt');assert shlex.split(command.decode())==[product]
     _,stdout=get(name+'-run-stdout.log');assert b'generated replay observations; status 0' in stdout
     continue
    status(name+'-baseline');_,command=get(name+'-baseline-command.txt');assert shlex.split(command.decode())==[product,'--fault-baseline']
    _,raw=get(name+'-baseline-stdout.log');matches=re.findall(rb'^NRG_FAULT_BASELINE calls=(\d+) peak=(\d+)$',raw,re.M);assert len(matches)==1;calls,peak=map(int,matches[0]);assert calls>0
    _,raw=get(name+'-fault-coverage.json');plan=json.loads(raw);assert plan['complete'] and plan['calls']==calls and plan['peak']==peak and plan['modes']==2
    cursor=0;recovered=0
    for begin,end,modes,count in plan['workers']:
     assert begin==cursor and begin<end<=calls and modes==2 and count==2*(end-begin)
     worker=f'{name}-fault-{begin:06d}-{end:06d}';status(worker)
     _,command=get(worker+'-command.txt');assert shlex.split(command.decode())==[product,'--fault-range',str(begin),str(end),str(calls)]
     _,raw=get(worker+'-stdout.log');matches=re.findall(rb'^NRG_FAULT_RANGE begin=(\d+) end=(\d+) calls=(\d+) modes=(\d+) recoveries=(\d+) peak=(\d+)$',raw,re.M);assert len(matches)==1 and tuple(map(int,matches[0]))==(begin,end,calls,2,count,peak)
     cursor=end;recovered+=count;workers+=1
    assert cursor==calls and recovered==plan['recoveries']==2*calls;recoveries+=recovered;coverage+=1
 result.append({'configuration':label,'artifact_report':path,'optimizations':opts,'products':products,'verified_child_statuses':statuses,'complete_fault_coverages':coverage,'fault_workers':workers,'recoveries':recoveries,'partial_history_scope':'O0 only from interrupted combined run; partial O2 excluded' if label.endswith('-O0') else None})
 print(label,products,statuses,workers,recoveries,flush=True)
out={'status':'PASS','sealed_git_pin':'3d4afccb726e18c8b59813c1de1b710e3b56636f','cases_per_mode_optimization':73,'configurations':result,'seconds':time.monotonic()-start,'scope':'Seven complete native configurations, with Linux Clang sanitized O0 and O2 independently attributed. Checks actual retained products, compile/link/verifier terminals, invoked executable identity, linked observation success, symbol exclusions and exact contiguous two-mode allocation recovery logs. Startup/emission/Wasm/package, normalized corpus equivalence and fresh current-main integration are separate obligations.'}
Path('/home/jkh/nanolang-qualification/llvm-root-native-coverage.json').write_text(json.dumps(out,indent=2)+'\n')
