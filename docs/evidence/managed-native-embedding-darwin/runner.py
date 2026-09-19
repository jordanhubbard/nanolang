import hashlib,json,os,pathlib,subprocess,time,sys
base=pathlib.Path(__file__).resolve().parent
root=base/'source'; evidence=base/'evidence'
evidence.mkdir(exist_ok=False)
def write(name,data): (evidence/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def output(args): return subprocess.check_output(args,text=True).strip()
sdk=output(['/usr/bin/xcrun','--show-sdk-path'])
otool=output(['/usr/bin/xcrun','--find','otool'])
cc='/opt/homebrew/bin/gcc-16'; clang='/opt/homebrew/opt/llvm/bin/clang'; python='/opt/homebrew/bin/python3'
env=os.environ.copy(); env.update(PATH=str(pathlib.Path(otool).parent)+':/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',SDKROOT=sdk,NMS_EMBED_GCC=cc,NMS_EMBED_CLANG=clang,NMS_EMBED_EVIDENCE=str(evidence/'artifacts'))
expected=json.loads((base/'source-expected.json').read_text())
def sources(): return {p:sha(root/p) for p in expected}
assert sources()==expected
cc1=output([cc,'-print-prog-name=cc1']); assembler=output(['/usr/bin/xcrun','--find','as']); linker=output(['/usr/bin/xcrun','--find','ld'])
paths=[cc,clang,python,cc1,assembler,linker,otool,'/usr/bin/xcrun',sdk+'/SDKSettings.json',sdk+'/usr/lib/libSystem.tbd',str(base/'runner.py')]
def tools(): return {p:{'resolved_path':str(pathlib.Path(p).resolve()),'sha256':sha(p)} for p in paths}
write('sources-before.json',sources()); write('tools-before.json',tools())
write('environment.json',{'source_pin':'5f988ed79dd1f343fd9dca3d9b40c928264e668c','uname':output(['/usr/bin/uname','-a']),'sdk':sdk,'selected_environment':{k:env[k] for k in ['PATH','SDKROOT','NMS_EMBED_GCC','NMS_EMBED_CLANG','NMS_EMBED_EVIDENCE']},'versions':{p:output([p,'--version']) for p in [cc,clang,python]},'otool':otool,'source_archive_sha256':sha(base/'source.tar')})
command=[python,'-m','unittest','discover','-s','tests','-p','test_managed_native_embedding.py','-v']
start=time.monotonic(); status=None
try:
 with (evidence/'gate.log').open('w') as log:
  result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  status=result.returncode
except subprocess.TimeoutExpired: status='timeout'
finally:
 write('result.json',{'command':command,'cwd':str(root),'status':status,'elapsed_seconds':time.monotonic()-start,'timeout_seconds':180})
 write('sources-after.json',sources());write('tools-after.json',tools())
 write('manifest.json',{str(p.relative_to(evidence)):sha(p) for p in sorted(evidence.rglob('*')) if p.is_file() and p.name!='manifest.json'})
assert sources()==expected
assert json.loads((evidence/'tools-before.json').read_text())==tools()
print((evidence/'gate.log').read_text()); print((evidence/'result.json').read_text())
sys.exit(0 if status==0 else 1)
