import hashlib,json,os,re,shlex,shutil,subprocess,sys,time,signal
from pathlib import Path
root=Path(sys.argv[1]).resolve();report=Path(sys.argv[2]).resolve();report.mkdir(parents=True,exist_ok=False);os.chdir(root)
artifacts=report/'artifacts';artifacts.mkdir();darwin=sys.platform=='darwin'
phases=set(os.environ.get('CARRIER_PHASES','').split(','))-{''}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def write(n,v):(report/n).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
files=(root/'.qualification-tracked').read_text().splitlines() if (root/'.qualification-tracked').exists() else subprocess.check_output(['git','ls-files','-z']).decode().split('\0')
files=[p for p in files if p and (root/p).is_file()]
def source():return {p:sha(root/p) for p in files}
def tools():
 if os.environ.get('CARRIER_HISTORY_BLOB'):
  actual=subprocess.check_output(['git','show','f1606e2c84e67491e9652a5bf71944d235216d95:src/nanoisa/nvm2c_file_private.c'])
  assert hashlib.sha256(actual).hexdigest()==sha(Path(os.environ['CARRIER_HISTORY_BLOB'])),'historical blob changed'
 paths={n:shutil.which(n) for n in ('make','python3','ar','ld','cc','gcc','clang')}
 paths.update(ordinary=shutil.which(os.environ['CC']),sanitizer=shutil.which(os.environ['CARRIER_SAN_CC']))
 if darwin:paths['libcrypto']='/opt/homebrew/opt/openssl@3/lib/libcrypto.dylib'
 paths.update(json.loads(os.environ.get('CARRIER_EXTRA_TOOLS','{}')))
 return {k:{'path':str(Path(v).resolve()),'sha256':sha(v)} for k,v in paths.items() if v}
def inventory(extra=()):
 entries={}
 paths=[]
 for directory in ('obj','bin'):
  if (root/directory).exists():paths+=list((root/directory).rglob('*'))
 for directory in extra:
  if Path(directory).is_dir():paths+=list(Path(directory).rglob('*'))
 for p in paths:
  if not p.is_file():continue
  if p.is_relative_to(root) and p.suffix not in ('.o','.a','.so','.dylib','.ll','.bc','.h','.json') and not os.access(p,os.X_OK):continue
  digest=sha(p);dest=artifacts/digest
  if not dest.exists():shutil.copyfile(p,dest)
  entries[str(p)]={'sha256':digest,'artifact':str(dest)}
 return entries
write('environment.json',{k:v for k,v in os.environ.items() if k in ('CC','CARRIER_SAN_CC','SDKROOT','LIBRARY_PATH','NMS_NATIVE_CLANG_FLAGS','NMS_RUNTIME_CLANG','PATH','CARRIER_PHASES','NANO_FILE_RUNTIME_CFLAGS','CARRIER_EXTRA_TOOLS','LSAN_OPTIONS','ASAN_OPTIONS','UBSAN_OPTIONS','GIT_DIR','GIT_WORK_TREE','CARRIER_HISTORY_BLOB','CARRIER_CONFIGURATION')})
for label,compiler in [('ordinary',os.environ['CC']),('sanitizer',os.environ['CARRIER_SAN_CC']),('clang',shutil.which('clang'))]:
 if compiler:
  p=subprocess.run([compiler,'--version'],capture_output=True,text=True);(report/(label+'-version.txt')).write_text(p.stdout+p.stderr)
if darwin:
 p=subprocess.run(['xcrun','--show-sdk-version'],capture_output=True,text=True);(report/'sdk-version.txt').write_text(p.stdout+p.stderr)
shutil.copyfile(__file__,report/'driver.py');results=[]
def run(label,cmd,extra=None):
 if phases and label not in phases:return
 before=source();toolbefore=tools();write(label+'-source-before.json',before);write(label+'-tools-before.json',toolbefore);write(label+'-inputs-before.json',inventory())
 env=dict(os.environ);env.update(extra or {})
 write(label+'-command.json',{'argv':cmd,'selectors':{k:v for k,v in env.items() if k.startswith(('NANO_','FILE_','WRAPPER_')) or k in ('CC','SDKROOT','LIBRARY_PATH','NMS_NATIVE_CLANG_FLAGS','LSAN_OPTIONS')}})
 start=time.monotonic();proc=None;timed_out=False;errors=[];signals=[];group_gone=None
 with (report/(label+'.log')).open('wb') as out:
  try:
   proc=subprocess.Popen(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   try:proc.wait(timeout=1800)
   except subprocess.TimeoutExpired:timed_out=True
  except Exception as error:errors.append(type(error).__name__+': '+str(error))
  finally:
   if proc is not None:
    def exists():
     try:os.killpg(proc.pid,0);return True
     except ProcessLookupError:return False
     except OSError as error:errors.append(str(error));return True
    for sig in (signal.SIGTERM,signal.SIGKILL):
     if not exists():break
     try:os.killpg(proc.pid,sig);signals.append(sig.name)
     except ProcessLookupError:pass
     except OSError as error:errors.append(str(error))
     deadline=time.monotonic()+5
     while time.monotonic()<deadline:
      proc.poll()
      if not exists():break
      time.sleep(.05)
    group_gone=not exists()
   code=proc.poll() if proc else None
   write(label+'-terminal.json',{'timeout':timed_out,'bound':1800,'returncode':code,'launched':proc is not None,'leader_reaped':code is not None,'group_disappeared':group_gone,'cleanup_signals':signals,'errors':errors})
 completed=subprocess.CompletedProcess(cmd,124 if timed_out else code if code is not None and not errors and group_gone else 125)
 result={'phase':label,'status':completed.returncode,'seconds':round(time.monotonic()-start,3)};results.append(result);write('results.json',results)
 after=source();toolafter=tools();write(label+'-source-after.json',after);write(label+'-tools-after.json',toolafter)
 text=(report/(label+'.log')).read_text();dirs=re.findall(r'I retain [^\n]*? artifacts at (\S+)',text)
 # Existing neighbors occasionally use a differently worded retention message.
 dirs+=re.findall(r'(/(?:tmp|var/folders)/[^\s\'\"]*nano-[^\s\'\"]+)',text)
 write(label+'-artifacts.json',inventory(dirs));print(json.dumps(result),flush=True)
 assert before==after and toolbefore==toolafter,'source or host tools changed during phase'
 if completed.returncode:sys.exit(completed.returncode)
probe=report/'providers.mk'
probe.write_text('''\n.PHONY: carrier-providers carrier-config
carrier-providers: $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o $(OBJ_DIR)/nanovirt/wrapper_gen.o $(OBJ_DIR)/nanovm/vmd_protocol.o $(OBJ_DIR)/nanovm/vmd_client.o
carrier-config:
\t@printf '%s\\n' 'ISA=$(NANOISA_OBJECTS) $(NANOISA_UTF8)' 'ALL=$(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o' 'LDFLAGS=$(LDFLAGS)' 'OPCODES=$(OBJ_DIR)/nanovirt/wrapper_gen.o $(OBJ_DIR)/nanoisa/nvm2llvm.o $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o'
''')
run('setup',['make','-f','Makefile.gnu','-f',str(probe),'-j2','CC='+os.environ['CC'],'carrier-providers','nvm2c','nvm2llvm','nvm2hl','nano_virt'])
run('configuration',['make','-s','-f','Makefile.gnu','-f',str(probe),'carrier-config','CC='+os.environ['CC']])
config=Path(os.environ.get('CARRIER_CONFIGURATION',str(report/'configuration.log'))).read_text();values=dict(line.split('=',1) for line in config.splitlines() if '=' in line)
filtered=' '.join(p for p in shlex.split(values['ISA']) if Path(p).stem not in ('file_flow','service_file_nominal','service_file_nominal_plan','nsi_file_plan'))
py=[sys.executable,'-m','unittest','-f','-v']
normal_flags=os.environ.get('NANO_FILE_RUNTIME_CFLAGS','')
clang_flags=normal_flags+(' --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13' if not darwin else '')
base={'FILE_RUNTIME_OBJECTS':values['ALL'],'FILE_RUNTIME_LDFLAGS':values['LDFLAGS']}
def runtime(label,test,compiler,flags,sanitize):
 run(label,py+[test],dict(base,NANO_FILE_RUNTIME_CC=compiler,NANO_FILE_RUNTIME_CFLAGS=flags,NANO_FILE_RUNTIME_SANITIZERS=str(sanitize)))
runtime('ordinary','tests.test_file_cyclic_runtime',os.environ['CC'],normal_flags,0)
runtime('clang-ordinary','tests.test_file_cyclic_runtime',shutil.which('clang'),clang_flags,0)
runtime('sanitizer','tests.test_file_cyclic_runtime',os.environ['CARRIER_SAN_CC'],normal_flags,1)
if not darwin:runtime('clang-sanitizer','tests.test_file_cyclic_runtime',shutil.which('clang'),clang_flags,1)
compiler=os.environ['CARRIER_SAN_CC']
run('core-values',py+['tests.test_nsi_file_values'],dict(NANO_FILE_VALUES_CC=compiler,NANO_FILE_VALUES_CFLAGS=normal_flags))
run('cyclic',py+['tests.test_file_cyclic'],dict(FILE_CYCLIC_OBJECTS=filtered,FILE_CYCLIC_LDFLAGS=values['LDFLAGS'],NANO_FILE_CYCLIC_CC=compiler,NANO_FILE_CYCLIC_CFLAGS=normal_flags+' -fsanitize=address,undefined -fno-omit-frame-pointer'))
run('cyclic-hosted',py+['tests.test_file_cyclic_hosted'],dict(FILE_CYCLIC_HOSTED_OBJECTS=values['ISA'],FILE_CYCLIC_HOSTED_LDFLAGS=values['LDFLAGS'],NANO_FILE_CYCLIC_HOSTED_CC=compiler,NANO_FILE_CYCLIC_HOSTED_CFLAGS=normal_flags,NANO_FILE_CYCLIC_HOSTED_SANITIZERS='1'))
runtime('private-vm','tests.test_file_private_vm',compiler,normal_flags,1)
runtime('private-native','tests.test_file_private_native',compiler,normal_flags,1)
# Public packaging rebuilds its providers. Keep it last and retain changed object endpoints explicitly.
runtime('public','tests.test_file_public',os.environ['CC'],normal_flags,0)

runtime('public-linked','tests.test_file_public.FilePublic.test_linked_native_corpus_and_isolated_refusals',os.environ['CC'],normal_flags,0)
