import hashlib,json,os,re,shlex,shutil,subprocess,sys,time,signal
from pathlib import Path
root=Path(sys.argv[1]).resolve();report=Path(sys.argv[2]).resolve();report.mkdir(parents=True,exist_ok=False);os.chdir(root)
artifacts=report/'artifacts';artifacts.mkdir();darwin=sys.platform=='darwin'
phases=set(os.environ.get('CARRIER_PHASES','').split(','))-{''}
assert not os.environ.get('LSAN_OPTIONS','');os.environ['LSAN_OPTIONS']=''
os.environ['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1';os.environ['UBSAN_OPTIONS']='halt_on_error=1'
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
 paths={n:shutil.which(n) for n in ('make','python3','ar','ld','cc','gcc','clang','pkg-config','ps')}
 paths['driver_python']=sys.executable
 paths['package_runner']=os.environ['PACKAGE_RUNNER']
 if darwin:
  paths['xcrun']='/usr/bin/xcrun'
  paths['apple_clang']=subprocess.check_output(['xcrun','--find','clang'],text=True).strip()
  paths['apple_ld']=subprocess.check_output(['xcrun','--find','ld'],text=True).strip()
 paths.update(ordinary=shutil.which(os.environ['CC']),sanitizer=shutil.which(os.environ['CARRIER_SAN_CC']))
 if darwin:paths['libcrypto']='/opt/homebrew/opt/openssl@3/lib/libcrypto.dylib'
 paths.update(json.loads(os.environ.get('CARRIER_EXTRA_TOOLS','{}')))
 for key,value in os.environ.items():
  if key.startswith('RECORD_LLVM_') and key not in ('RECORD_LLVM_C_BASELINE','RECORD_LLVM_IR_ASAN','RECORD_LLVM_NATIVE_OPTIMIZATIONS'):
   executable=shlex.split(value)[0];paths[key]=shutil.which(executable) or executable
 return {k:{'path':str(Path(v).resolve()),'sha256':sha(v)} for k,v in paths.items() if v}
def inventory(extra=()):
 entries={}
 paths=[]
 for directory in ('obj','bin','lib'):
  if (root/directory).exists():paths+=list((root/directory).rglob('*'))
 for directory in extra:
  if Path(directory).is_dir():paths+=list(Path(directory).rglob('*'))
 for p in paths:
  if not p.is_file():continue
  if p.is_relative_to(root) and p.suffix not in ('.o','.a','.so','.dylib','.ll','.bc','.h','.json') and not os.access(p,os.X_OK):continue
  digest=sha(p);dest=artifacts/digest
  if not dest.exists():
   shared=None
   for prior in ():
    candidate=Path('/tmp/nanolang-record-vm-'+prior+'-'+('puck' if darwin else 'linux'))/'artifacts'/digest
    if candidate.is_file() and sha(candidate)==digest:shared=candidate;break
   if shared is not None:os.link(shared,dest)
   elif not p.is_relative_to(root):os.link(p,dest)
   else:shutil.copyfile(p,dest)
  entries[str(p)]={'sha256':digest,'artifact':str(dest)}
 return entries
write('environment.json',{k:v for k,v in os.environ.items() if k in ('CC','CARRIER_SAN_CC','SDKROOT','LIBRARY_PATH','NMS_NATIVE_CLANG_FLAGS','PATH','CARRIER_PHASES','NANO_FILE_RUNTIME_CFLAGS','CARRIER_EXTRA_TOOLS','LSAN_OPTIONS','ASAN_OPTIONS','UBSAN_OPTIONS','TMPDIR','RECORD_LLVM_NATIVE_OPTIMIZATIONS')})
for label,compiler in [('ordinary',os.environ['CC']),('sanitizer',os.environ['CARRIER_SAN_CC']),('clang',shutil.which('clang'))]:
 if compiler:
  p=subprocess.run([compiler,'--version'],capture_output=True,text=True);(report/(label+'-version.txt')).write_text(p.stdout+p.stderr)
if darwin:
 p=subprocess.run(['xcrun','--show-sdk-version'],capture_output=True,text=True);(report/'sdk-version.txt').write_text(p.stdout+p.stderr)
shutil.copyfile(__file__,report/'driver.py');results=[]
baseline=Path(os.environ['RECORD_LLVM_C_BASELINE']);baseline_before={str(p):sha(p) for p in baseline.glob('product-*.c')};assert len(baseline_before)==146
write('c-baseline-before.json',baseline_before)
def run(label,cmd,extra=None):
 if phases and label not in phases:return
 free=shutil.disk_usage(report).free;write(label+'-space-before.json',{'free_bytes':free,'minimum_bytes':2147483648})
 if free<2147483648:write(label+'-terminal.json',{'launched':False,'reason':'capacity preflight','free_bytes':free});raise SystemExit(125)
 before=source();toolbefore=tools();write(label+'-source-before.json',before);write(label+'-tools-before.json',toolbefore);write(label+'-inputs-before.json',inventory())
 env=dict(os.environ);env.update(extra or {})
 write(label+'-command.json',{'argv':cmd,'selectors':{k:v for k,v in env.items() if k.startswith(('NANO_','FILE_','WRAPPER_','ORDINARY_','DECLARATION_','RECORD_ARRAY_','RECORD_GENERATED_','RECORD_LLVM_')) or k in ('CC','SDKROOT','LIBRARY_PATH','NMS_NATIVE_CLANG_FLAGS','LSAN_OPTIONS')}})
 phase_bound=28800 if label=='wasm' else 14400 if label.endswith('-native') else 1800
 start=time.monotonic();proc=None;timed_out=False;errors=[];signals=[];group_gone=None;owned={};nested_signals=[];nested_remaining=[]
 with (report/(label+'.log')).open('wb') as out:
  try:
   proc=subprocess.Popen(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
   def process_rows():
    rows={}
    for line in subprocess.check_output(['ps','-axo','pid=,ppid=,pgid=,lstart=,args='],text=True).splitlines():
     fields=line.strip().split(None,8)
     if len(fields)==9:rows[int(fields[0])]={'parent':int(fields[1]),'group':int(fields[2]),'started':' '.join(fields[3:8]),'command':fields[8]}
    return rows
   deadline=time.monotonic()+phase_bound
   while proc.poll() is None:
    rows=process_rows();descendants={proc.pid}
    for _ in range(len(rows)+1):
     more={pid for pid,row in rows.items() if row['parent'] in descendants}
     if more<=descendants:break
     descendants.update(more)
    for pid in descendants:
     if pid in rows:owned[pid]=rows[pid]
    free=shutil.disk_usage(report).free
    if free<1610612736:
     write(label+'-capacity-stop.json',{'free_bytes':free,'minimum_bytes':1610612736});errors.append('active output capacity guard');break
    remaining=deadline-time.monotonic()
    if remaining<=0:timed_out=True;break
    try:proc.wait(timeout=min(1,remaining))
    except subprocess.TimeoutExpired:pass
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
    # Nested fixture commands have distinct groups: preserve and clean those too.
    for sig in (signal.SIGTERM,signal.SIGKILL):
     rows=process_rows()
     groups={row['group'] for pid,row in rows.items() if pid in owned and row['started']==owned[pid]['started'] and row['group']!=proc.pid}
     for group in groups:
      try:os.killpg(group,sig);nested_signals.append({'group':group,'signal':sig.name})
      except ProcessLookupError:pass
      except OSError as error:errors.append(str(error))
     if groups:time.sleep(.2)
    limit=time.monotonic()+5
    while True:
     rows=process_rows();nested_remaining=[{'pid':pid,**row} for pid,row in rows.items() if pid in owned and row['started']==owned[pid]['started'] and pid!=proc.pid]
     if not nested_remaining or time.monotonic()>=limit:break
     time.sleep(.05)
    write(label+'-nested-processes.json',{'owned':owned,'cleanup':nested_signals,'remaining':nested_remaining})
    if nested_remaining:errors.append('nested processes remain after bounded cleanup')
   code=proc.poll() if proc else None
   write(label+'-terminal.json',{'timeout':timed_out,'bound':phase_bound,'returncode':code,'launched':proc is not None,'leader_reaped':code is not None,'group_disappeared':group_gone,'cleanup_signals':signals,'errors':errors})
 completed=subprocess.CompletedProcess(cmd,124 if timed_out else code if code is not None and not errors and group_gone else 125)
 result={'phase':label,'status':completed.returncode,'seconds':round(time.monotonic()-start,3)};results.append(result);write('results.json',results)
 after=source();toolafter=tools();write(label+'-source-after.json',after);write(label+'-tools-after.json',toolafter)
 text=(report/(label+'.log')).read_text(errors='replace');dirs=re.findall(r'I retain [^\n]*? artifacts at (\S+)',text)
 # Existing neighbors occasionally use a differently worded retention message.
 dirs+=re.findall(r'(/(?:tmp|var/folders)/[^\s\'\"]*nano-[^\s\'\"]+)',text)
 write(label+'-artifacts.json',inventory([*dirs,report/'controls',report/'package-artifacts']));print(json.dumps(result),flush=True)
 assert before==after and toolbefore==toolafter,'source or host tools changed during phase'
 if completed.returncode:sys.exit(completed.returncode)

os.environ['LSAN_OPTIONS']=''
controls=report/'controls';controls.mkdir()
probe=report/'providers.mk'
probe.write_text("\n.PHONY: array-providers array-config\narray-providers: $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)\narray-config:\n\t@printf '%s\\n' 'ISA=$(NANOISA_OBJECTS) $(NANOISA_UTF8)' 'VM=$(sort $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS))' 'CFLAGS=$(CFLAGS)' 'LDFLAGS=$(LDFLAGS)'\n")
run('setup',['make','-f','Makefile.gnu','-f',str(probe),'-j2','CC='+os.environ['CC'],'array-providers','nano_vm','nvm2c','managed-runtime-package'])
run('configuration',['make','-s','-f','Makefile.gnu','-f',str(probe),'CC='+os.environ['CC'],'array-config'])
values=dict(line.split('=',1) for line in (report/'configuration.log').read_text().splitlines() if '=' in line)
objects=shlex.split(values['ISA']);links=shlex.split(values['LDFLAGS'])
flags=values['CFLAGS']+' -ffp-contract=off -fno-fast-math'
base=dict(RECORD_GENERATED_VM_OBJECTS=values['VM'],RECORD_GENERATED_QUERY_OBJECTS=values['ISA'],RECORD_GENERATED_LDFLAGS=values['LDFLAGS'])
py=[sys.executable,'-m','unittest','-f','-v']
clang=shutil.which('clang');clang_flags='' if darwin else ' --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'
sanitize=' -fsanitize=address,undefined -fno-omit-frame-pointer'
run('discovery',[sys.executable,'-c',"import unittest;from tests import test_record_array_llvm as m;s=unittest.defaultTestLoader.loadTestsFromModule(m);print(s.countTestCases());assert s.countTestCases()==4"])
ordinary=dict(base,RECORD_GENERATED_CC=os.environ['CC'],RECORD_GENERATED_CFLAGS=flags,RECORD_LLVM_IR_ASAN='0')
run('generated-discovery',[sys.executable,'-c',"import unittest;from tests import test_record_array_generated as m;s=unittest.defaultTestLoader.loadTestsFromModule(m);print(s.countTestCases());assert s.countTestCases()==2"])
run('generated-ordinary-native',py+['tests.test_record_array_generated'],ordinary)
run('generated-clang-native',py+['tests.test_record_array_generated'],dict(ordinary,RECORD_GENERATED_CC=clang,RECORD_GENERATED_CFLAGS=flags+clang_flags))
run('parity',py+['tests.test_record_array_llvm.RecordArrayLLVM.test_00_factored_c_bytes'],ordinary)
configs=[('ordinary',os.environ['CC'],flags,False),('clang-ordinary',clang,flags+clang_flags,False)]
for label,cc,cflags,asan in configs:
 selected=dict(base,RECORD_GENERATED_CC=cc,RECORD_GENERATED_CFLAGS=cflags,RECORD_LLVM_IR_ASAN=str(int(asan)))
 run(label+'-emission',py+['tests.test_record_array_llvm.RecordArrayLLVM.test_01_emission_bounds'],selected)
 run(label+'-native',py+['tests.test_record_array_llvm.RecordArrayLLVM.test_02_native_corpus'],selected)
run('wasm',py+['tests.test_record_array_llvm.RecordArrayLLVM.test_03_wasm_corpus'],ordinary)
run('old-private-vm',py+['tests.test_record_array_vm'],dict(RECORD_ARRAY_VM_OBJECTS=values['VM'],RECORD_ARRAY_VM_LDFLAGS=values['LDFLAGS'],RECORD_ARRAY_VM_CC=os.environ['CC'],RECORD_ARRAY_VM_CFLAGS=flags))
for name in ('record_array_global_flow','record_array_execution','record_array_origins'):
 run(name,py+['tests.test_'+name],dict(RECORD_ARRAY_OBJECTS=values['ISA'],RECORD_ARRAY_LDFLAGS=values['LDFLAGS'],RECORD_ARRAY_CC=os.environ['CC'],RECORD_ARRAY_CFLAGS=flags))

package_runner=Path(os.environ['PACKAGE_RUNNER']);shutil.copy2(package_runner,controls/'package-retain.py')
run('package',[sys.executable,str(controls/'package-retain.py')],dict(PACKAGE_ARTIFACTS=str(report/'package-artifacts'),PYTHONPATH=str(root)))

write("c-baseline-after.json",{p:sha(p) for p in baseline_before});assert baseline_before=={p:sha(p) for p in baseline_before}
