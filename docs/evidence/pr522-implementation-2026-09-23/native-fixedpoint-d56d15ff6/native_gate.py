"""I retain bounded native compiler generations from one clean source pin."""
import hashlib,json,os,pathlib,signal,subprocess,time
root=pathlib.Path('/qualification-final-d56d15ff6')
out=pathlib.Path('/evidence-native-d56d15ff6-retry')
out.mkdir(exist_ok=False)
expected='d56d15ff6934dcd4872aa0f90bfe7ea828cf79fa'
env=os.environ.copy()
env.update(CC='cc',NANO_CC='cc',NANO_AS_CAPTURE_HELPER=str(root/'bin/nano_as_capture.so'),NANO_MODULE_PATH=str(root/'modules'))
manifest={'source_commit':expected,'stages':{},'stage_timeout_seconds':1800,'complete':False,'boundary':'I compare raw modules from two standalone native compiler generations. I preserve the existing default shadow deadline and record this 16 GiB Linux VM separately from historical hosts.'}
def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def git(*args): return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def digest(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(label,args):
 start=time.monotonic()
 with (out/(label+'.log')).open('wb') as log:
  p=subprocess.Popen([str(x) for x in args],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try: code=p.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGKILL);p.wait()
   manifest['stages'][label]={'argv':list(map(str,args)),'timeout':True,'seconds':time.monotonic()-start};save();raise
 manifest['stages'][label]={'argv':list(map(str,args)),'exit_code':code,'seconds':time.monotonic()-start};save()
 if code: raise RuntimeError(label+' failed; I retain its log')
def hosts(path,label):
 dump=subprocess.check_output([str(root/'bin/nanoisa'),'dump',str(path)],cwd=root,text=True)
 (out/(label+'.nasm')).write_text(dump)
 import shlex
 libraries={shlex.split(line)[1] for line in dump.splitlines() if line.startswith('.import ') and shlex.split(line)[1]}
 assert all(pathlib.Path(p).is_absolute() for p in libraries)
 return {p:digest(p) for p in sorted(libraries)}
def verify(path,label):run(label,[root/'bin/nano_vm','--verify-only',path])
def native(path,label):
 c=out/(label+'.c');exe=out/(label+'-native')
 run(label+'-translate',[root/'bin/nvm2c',path,'-o',c])
 run(label+'-native-build',['cc','-std=c11','-Wall','-Wextra','-Werror','-O0',c,'-o',exe,root/'bin/nano_aot_runtime.o','-lm','-Wl,--export-dynamic','-ldl'])
 run(label+'-help',[exe,'--help'])
 run(label+'-libraries',['ldd',exe])
 assert 'libnanovm' not in (out/(label+'-libraries.log')).read_text().lower()
 return exe
assert git('rev-parse','HEAD')==expected and not git('status','--porcelain')
save()
run('providers',['make','-j2','bin/nanoc_c','nvm2c','nvm2c-runtime'])
assert not git('status','--porcelain')
tools={str(root/p):digest(root/p) for p in ['bin/nanoc_c','bin/nvm2c','bin/nano_vm','bin/nanoisa','bin/nano_as_capture.so','bin/nano_aot_runtime.o']}
manifest['providers']=tools;save()
source=root/'src_nano/nanoc_v06.nano'
seed=out/'seed'
run('seed-build',[root/'bin/nanoc_c',source,'-o',seed])
initial=out/'initial.nvm'
run('initial-emission',[seed,source,'--emit-nvm','-o',initial]);verify(initial,'initial-verify')
closure=hosts(initial,'initial');manifest['host_libraries']=closure;save()
compiler=native(initial,'initial')
first=out/'stage1.nvm';second=out/'stage2.nvm'
for label,target in [('stage1',first),('stage2',second)]:
 run(label,[compiler,source,'--emit-nvm','-o',target]);verify(target,label+'-verify')
 assert hosts(target,label)==closure
 manifest['stages'][label].update(sha256=digest(target),bytes=target.stat().st_size);save()
 if label=='stage1':compiler=native(target,label)
assert first.read_bytes()==second.read_bytes()
manifest['raw_stage1_stage2_equal']=True;save()
# I translate the final compared generation too, then exercise its product.
compiler=native(second,'stage2')
hello=out/'hello.nvm'
run('hello-compile',[compiler,root/'examples/language/nl_hello.nano','--emit-nvm','-o',hello])
verify(hello,'hello-verify');run('hello-execute',[root/'bin/nano_vm',hello])
assert {p:digest(p) for p in closure}==closure
assert {p:digest(p) for p in tools}==tools
assert git('rev-parse','HEAD')==expected and not git('status','--porcelain')
manifest['complete']=True;save()
