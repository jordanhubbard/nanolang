from pathlib import Path
import hashlib, json, os, platform, subprocess, time
root=Path('/Users/jordanh/Src/nanolang')
out=root/'docs/evidence/socket-tcp-20261009'
env=dict(os.environ, ASAN_OPTIONS='detect_leaks=1', PATH='/opt/homebrew/opt/llvm/bin:'+os.environ['PATH'])
records=[]
for name,cc,flags in [('llvm-sanitizers','/opt/homebrew/opt/llvm/bin/clang',['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-sanitize-recover=all']),('apple-strict','/usr/bin/clang',['-O2']),('gcc-strict','/opt/homebrew/bin/gcc-16',['-O2'])]:
 build=Path('/private/tmp/nanolang-tcp-'+name);build.mkdir(exist_ok=True)
 tools=subprocess.run([cc,'--version'],capture_output=True,text=True,check=True).stdout
 (out/(name+'-tool.txt')).write_text(tools)
 for linkage,sources in [('instrumented',['tests/test_nsi_socket.c']),('linked',['tests/test_nsi_socket_linked.c','src/nsi_socket.c','src/nsi_cap.c'])]:
  binary=build/linkage
  for phase,args in [('compile',[cc,'-std=c11','-D_DEFAULT_SOURCE','-D_DARWIN_C_SOURCE','-Wall','-Wextra','-Werror',*flags,*sources,'-o',str(binary)]),('execute',[str(binary)])]:
   log=out/(name+'-'+linkage+'-'+phase+'.log');start=time.monotonic()
   with log.open('w') as stream:
    p=subprocess.run(args,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
   record={'name':name,'linkage':linkage,'phase':phase,'args':args,'returncode':p.returncode,'seconds':time.monotonic()-start,'log':log.name}
   if binary.exists():record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
   records.append(record);(out/'runs.json').write_text(json.dumps({'host':platform.platform(),'runs':records},indent=2)+'\n')
   print(name,linkage,phase,p.returncode,flush=True)
   if p.returncode:raise SystemExit(p.returncode)
args=['make','-f','Makefile.gnu','CC=/opt/homebrew/opt/llvm/bin/clang','CFLAGS=-std=c11 -g -O1 -Isrc -D_GNU_SOURCE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all','LDFLAGS=-fsanitize=address,undefined','test-nsi-file','test-nsi-cap']
start=time.monotonic()
with (out/'adjacency-sanitizers.log').open('w') as stream:p=subprocess.run(args,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
records.append({'name':'adjacency','args':args,'returncode':p.returncode,'seconds':time.monotonic()-start,'log':'adjacency-sanitizers.log'})
(out/'runs.json').write_text(json.dumps({'host':platform.platform(),'runs':records},indent=2)+'\n')
print('adjacency',p.returncode,flush=True)
assert p.returncode == 0
inputs=json.loads((out/'inputs.json').read_text())
assert inputs == {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in inputs}
