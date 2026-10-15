from pathlib import Path
import hashlib,json,os,platform,subprocess,time
root=Path('/Users/jordanh/Src/nanolang');out=root/'docs/evidence/socket-catalog-20261009'
records=[]
env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
base=['-std=c11','-D_DEFAULT_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror']
for name,cc,flags in [('llvm-sanitizers','/opt/homebrew/opt/llvm/bin/clang',['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-sanitize-recover=all']),('apple-strict','/usr/bin/clang',['-O2']),('gcc-strict','/opt/homebrew/bin/gcc-16',['-O2'])]:
 build=Path('/private/tmp/nanolang-socket-catalog-'+name);build.mkdir(exist_ok=True)
 (out/(name+'-tool.txt')).write_text(subprocess.run([cc,'--version'],capture_output=True,text=True,check=True).stdout)
 for subject in ['socket-plan','file-plan','socket-values']:
  for linkage in ['instrumented','linked']:
   label=name+'-'+subject+'-'+linkage;binary=build/(subject+'-'+linkage)
   if subject.endswith('-plan'):
    prefix=subject.split('-')[0];sources=['tests/test_nsi_'+prefix+'_plan.c','src/nsi.c','src/utf8.c','src/cJSON.c']
    if linkage=='instrumented':sources.insert(0,'-D'+prefix.upper()+'_PLAN_INSTRUMENT')
    else:sources.insert(1,'src/nsi_'+prefix+'_plan.c')
    run_args=[str(binary),'tests/fixtures/nsi_'+prefix+'_plan.json']
   else:
    sources=['tests/test_nsi_socket_values.c'] if linkage=='instrumented' else ['tests/test_nsi_socket_values_linked.c','src/nsi_socket_values.c','src/nsi_socket.c','src/nsi_cap.c']
    run_args=[str(binary)]
   for phase,args in [('compile',[cc,*base,*flags,*sources,'-o',str(binary)]),('execute',run_args)]:
    log=out/(label+'-'+phase+'.log');start=time.monotonic()
    with log.open('w') as stream:p=subprocess.run(args,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=90)
    record={'name':name,'subject':subject,'linkage':linkage,'phase':phase,'args':args,'returncode':p.returncode,'seconds':time.monotonic()-start,'log':log.name}
    if binary.exists():record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    records.append(record);(out/'runs.json').write_text(json.dumps({'host':platform.platform(),'runs':records},indent=2)+'\n')
    print(label,phase,p.returncode,flush=True)
    if p.returncode:raise SystemExit(p.returncode)
inputs=json.loads((out/'inputs.json').read_text());assert inputs=={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs}
