import hashlib,json,os,re,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();report=Path(sys.argv[2]).resolve();report.mkdir(parents=True,exist_ok=False);os.chdir(root)
store=report/'artifacts';store.mkdir();darwin=sys.platform=='darwin'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(n,v):(report/n).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
files=subprocess.check_output(['git','ls-files','-z']).decode().split('\0');files=[p for p in files if p and (root/p).is_file()]
def sources():return {p:sha(root/p) for p in files}
def tools():
 names=['make','python3','ar','nm','ld','pkg-config','cc','gcc','clang']
 values={n:shutil.which(n) for n in names};values.update(ordinary=os.environ['PUBLIC_CC'],clang_selected=os.environ['PUBLIC_CLANG'])
 values.update(json.loads(os.environ.get('PUBLIC_EXTRA_TOOLS','{}')))
 return {k:{'path':str(Path(p).resolve()),'sha256':sha(p)} for k,p in values.items() if p and Path(p).is_file()}
def inventory(extra=()):
 paths=[]
 for d in [root/'obj',root/'bin',root/'lib',*(Path(x) for x in extra)]:
  if d.is_dir():paths.extend(p for p in d.rglob('*') if p.is_file())
 out={}
 for p in paths:
  digest=sha(p);dest=store/digest
  if not dest.exists():shutil.copyfile(p,dest)
  out[str(p)]={'sha256':digest,'artifact':str(dest)}
 return out
write('source-pin.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'root':str(root)})
write('environment.json',{k:v for k,v in os.environ.items() if k in ('PATH','PUBLIC_CC','PUBLIC_CLANG','PUBLIC_PHASES','PUBLIC_EXTRA_TOOLS','SDKROOT','LIBRARY_PATH','CPATH','NMS_NATIVE_CLANG_FLAGS')})
shutil.copyfile(__file__,report/'driver.py');results=[]
phases=set(os.environ.get('PUBLIC_PHASES','').split(','))-{''}
def run(label,argv,env,limit=3600):
 if phases and label not in phases:return
 before=sources();tb=tools();write(label+'-source-before.json',before);write(label+'-tools-before.json',tb);write(label+'-providers-before.json',inventory())
 write(label+'-command.json',{'argv':argv,'environment':{k:v for k,v in env.items() if k.startswith(('NANO_','FILE_','PUBLIC_')) or k in ('CC','SDKROOT','LIBRARY_PATH','LSAN_OPTIONS','ASAN_OPTIONS','UBSAN_OPTIONS')}})
 started=time.monotonic();proc=None;problem=None;cleanup=[]
 try:
  with (report/(label+'.log')).open('w') as log:
   proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   proc.wait(timeout=limit)
 except BaseException as e:problem=repr(e)
 finally:
  if proc:
   for sig in (signal.SIGTERM,signal.SIGKILL):
    try:os.killpg(proc.pid,sig)
    except ProcessLookupError:pass
    except OSError as e:cleanup.append(repr(e))
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:cleanup.append('wait expired '+sig.name)
  status=proc.returncode if proc else 127
  if problem:status=124 if 'TimeoutExpired' in problem else 127
  result={'phase':label,'status':status,'seconds':round(time.monotonic()-started,3),'exception':problem,'cleanup_errors':cleanup};results.append(result);write('results.json',results)
  after=sources();ta=tools();write(label+'-source-after.json',after);write(label+'-tools-after.json',ta)
  text=(report/(label+'.log')).read_text(errors='replace') if (report/(label+'.log')).exists() else ''
  dirs=re.findall(r'I retain [^\n]*? artifacts at (\S+)',text)
  write(label+'-artifacts.json',inventory(dirs));print(json.dumps(result),flush=True)
 if before!=after or tb!=ta:raise RuntimeError('source/tool drift')
 if status or cleanup:sys.exit(status or 1)
probe=report/'providers.mk';probe.write_text('''\n.PHONY: public-providers public-config
public-providers: $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o $(OBJ_DIR)/nanovirt/wrapper_gen.o $(OBJ_DIR)/nanovm/vmd_protocol.o $(OBJ_DIR)/nanovm/vmd_client.o
public-config:
\t@printf '%s\\n' 'ALL=$(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o' 'ISA=$(NANOISA_OBJECTS) $(NANOISA_UTF8)' 'OPCODES=$(OBJ_DIR)/nanovirt/wrapper_gen.o $(OBJ_DIR)/nanoisa/nvm2llvm.o $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o' 'LDFLAGS=$(LDFLAGS)'
''')
env=dict(os.environ,LSAN_OPTIONS='',ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
for label,path in [('ordinary',env['PUBLIC_CC']),('clang',env['PUBLIC_CLANG'])]:
 r=subprocess.run([path,'--version'],capture_output=True,text=True);(report/(label+'-version.txt')).write_text(r.stdout+r.stderr)
run('ordinary-neighbor-setup',['make','-B','-f','Makefile.gnu','-f',str(probe),'-j2','CC='+env['PUBLIC_CC'],'public-providers','nvm2c','nvm2llvm','nvm2hl','nano_virt'],env,1200)
cfg=subprocess.check_output(['make','-s','-f','Makefile.gnu','-f',str(probe),'public-config'],text=True);(report/'provider-config.txt').write_text(cfg);values=dict(x.split('=',1) for x in cfg.splitlines() if '=' in x)
cc=env['PUBLIC_CLANG'] if darwin else env['PUBLIC_CC']
flags=os.environ.get('PUBLIC_DARWIN_CFLAGS','') if darwin else ''
base=dict(env,FILE_OPCODE_OBJECTS=values['OPCODES'],FILE_OPCODE_LDFLAGS=values['LDFLAGS'],NANO_FILE_OPCODE_TEST_CC=cc,NANO_FILE_OPCODE_TEST_CFLAGS=flags,NANO_FILE_HOST_GRANT_CC=cc,NANO_FILE_HOST_GRANT_CFLAGS=flags+' -O1 -g -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer')
run('opcode-refusal',[sys.executable,'-m','unittest','-f','-v','tests.test_file_opcodes'],base)
run('host-grant',[sys.executable,'-m','unittest','-f','-v','tests.test_file_host_grant'],base)
wrapper_code="import tempfile,unittest; tempfile.TemporaryDirectory.cleanup=lambda self:(self._finalizer.detach(),print('I retain wrapper artifacts at '+self.name,flush=True)); unittest.main(module='tests.test_wrapper_publication',verbosity=2,failfast=True)"
run('ordinary-wrapper',[sys.executable,'-c',wrapper_code],base)
