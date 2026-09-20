import hashlib,json,os,re,shlex,shutil,subprocess,sys,time,signal
from pathlib import Path
root=Path(sys.argv[1]).resolve();report=Path(sys.argv[2]).resolve();report.mkdir(parents=True,exist_ok=False);os.chdir(root)
artifacts=report/'artifacts';artifacts.mkdir();darwin=sys.platform=='darwin'
phases={'setup','configuration','discovery','ordinary','cyclic','cyclic-hosted','indirect-flow','public','archive-boundary'}
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
 paths={n:shutil.which(n) for n in ('make','python3','ar','ld','cc','gcc','clang','git','nm')}
 paths.update(ordinary=shutil.which(os.environ['CC']),sanitizer=shutil.which(os.environ['CARRIER_SAN_CC']))
 paths['driver_python']=sys.executable
 if darwin:paths['libcrypto']='/opt/homebrew/opt/openssl@3/lib/libcrypto.dylib'
 paths.update(json.loads(os.environ.get('CARRIER_EXTRA_TOOLS','{}')))
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
  if not dest.exists():shutil.copyfile(p,dest)
  entries[str(p)]={'sha256':digest,'artifact':str(dest)}
 return entries
write('environment.json',{k:v for k,v in os.environ.items() if k in ('CC','CARRIER_SAN_CC','SDKROOT','LIBRARY_PATH','NMS_NATIVE_CLANG_FLAGS','NMS_RUNTIME_CLANG','PATH','CARRIER_PHASES','NANO_FILE_RUNTIME_CFLAGS','CARRIER_EXTRA_TOOLS','LSAN_OPTIONS','ASAN_OPTIONS','UBSAN_OPTIONS')})
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
carrier-providers: $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o $(OBJ_DIR)/nanovirt/wrapper_gen.o $(OBJ_DIR)/nanovm/vmd_protocol.o $(OBJ_DIR)/nanovm/vmd_client.o $(FILE_CYCLIC_PRIVATE_PROVIDERS) $(NANOISA_UTF8)
carrier-config:
\t@printf '%s\\n' 'ISA=$(NANOISA_OBJECTS) $(NANOISA_UTF8)' 'ALL=$(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o' 'LDFLAGS=$(LDFLAGS)' 'NATIVE=$(FILE_CYCLIC_PRIVATE_PROVIDERS) $(NANOISA_UTF8)' 'PUBLIC=$(FILE_PUBLIC_OBJECTS)' 'OPCODES=$(OBJ_DIR)/nanovirt/wrapper_gen.o $(OBJ_DIR)/nanoisa/nvm2llvm.o $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(OBJ_DIR)/nsi.o'
''')
run('setup',['make','-f','Makefile.gnu','-f',str(probe),'-j2','CC='+os.environ['CC'],'carrier-providers','nvm2c','nvm2llvm','nvm2hl','nano_virt','file-public-runtime'])
run('configuration',['make','-s','-f','Makefile.gnu','-f',str(probe),'carrier-config','CC='+os.environ['CC']])
config=(report/'configuration.log').read_text();values=dict(line.split('=',1) for line in config.splitlines() if '=' in line)
filtered=' '.join(p for p in shlex.split(values['ISA']) if Path(p).stem not in ('file_flow','service_file_nominal','service_file_nominal_plan','nsi_file_plan'))
py=[sys.executable,'-m','unittest','-f','-v']
normal_flags=os.environ.get('NANO_FILE_RUNTIME_CFLAGS','')
clang_flags=normal_flags+(' --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13' if not darwin else '')
base={'FILE_RUNTIME_OBJECTS':values['ALL'],'FILE_RUNTIME_LDFLAGS':values['LDFLAGS'],'FILE_CYCLIC_NATIVE_LINK_OBJECTS':values['NATIVE']}
def runtime(label,test,compiler,flags,sanitize):
 run(label,py+[test],dict(base,NANO_FILE_RUNTIME_CC=compiler,NANO_FILE_RUNTIME_CFLAGS=flags,NANO_FILE_RUNTIME_SANITIZERS=str(sanitize)))
run('discovery',[sys.executable,'-c',"import unittest;from tests import test_file_cyclic_dispatch as m;s=unittest.defaultTestLoader.loadTestsFromModule(m);print(s.countTestCases());assert s.countTestCases()==2"],base)
runtime('ordinary','tests.test_file_cyclic_dispatch',os.environ['CC'],normal_flags,0)
runtime('clang-ordinary','tests.test_file_cyclic_dispatch',shutil.which('clang'),clang_flags,0)
runtime('sanitizer','tests.test_file_cyclic_dispatch',os.environ['CARRIER_SAN_CC'],normal_flags,1)
if not darwin:runtime('clang-sanitizer','tests.test_file_cyclic_dispatch',shutil.which('clang'),clang_flags,1)
compiler=os.environ['CC']
runtime('carrier','tests.test_file_cyclic_runtime',compiler,normal_flags,1)
run('core-values',py+['tests.test_nsi_file_values'],dict(NANO_FILE_VALUES_CC=compiler,NANO_FILE_VALUES_CFLAGS=normal_flags))
run('cyclic',py+['tests.test_file_cyclic'],dict(FILE_CYCLIC_OBJECTS=filtered,FILE_CYCLIC_LDFLAGS=values['LDFLAGS'],NANO_FILE_CYCLIC_CC=compiler,NANO_FILE_CYCLIC_CFLAGS=normal_flags))
run('cyclic-hosted',py+['tests.test_file_cyclic_hosted'],dict(FILE_CYCLIC_HOSTED_OBJECTS=values['ISA'],FILE_CYCLIC_HOSTED_LDFLAGS=values['LDFLAGS'],NANO_FILE_CYCLIC_HOSTED_CC=compiler,NANO_FILE_CYCLIC_HOSTED_CFLAGS=normal_flags,NANO_FILE_CYCLIC_HOSTED_SANITIZERS='0'))
runtime('private-vm','tests.test_file_private_vm',compiler,normal_flags,1)
runtime('private-native','tests.test_file_private_native',compiler,normal_flags,1)
run('indirect-flow',py+['tests.test_file_indirect_flow'],dict(FILE_INDIRECT_FLOW_OBJECTS=filtered,FILE_INDIRECT_FLOW_LDFLAGS=values['LDFLAGS'],NANO_FILE_INDIRECT_FLOW_CC=compiler,NANO_FILE_INDIRECT_FLOW_CFLAGS=normal_flags))
# The archived Darwin tree supplies history only to this neighbor's read-only git show.
history=Path('/Users/jkh/Src/nanolang/.git') if darwin else root/'.git'
history_env=dict(os.environ)
if darwin:history_env.update(GIT_DIR=str(history),GIT_WORK_TREE=str(root))
commit='f1606e2c84e67491e9652a5bf71944d235216d95'
blob=subprocess.check_output(['git','rev-parse',commit+':src/nanoisa/nvm2c_file_private.c'],env=history_env,text=True).strip()
assert blob=='732f38decf290ab7e3288d7f77421848aa235c24'
write('public-history-before.json',{'git_dir':str(history),'commit':commit,'blob':blob,'git':tools()['git']})
public_extra=dict(base,NANO_FILE_RUNTIME_CC=os.environ['CC'],NANO_FILE_RUNTIME_CFLAGS=normal_flags,NANO_FILE_RUNTIME_SANITIZERS='0')
if darwin:public_extra.update(GIT_DIR=str(history),GIT_WORK_TREE=str(root))
run('public',py+['tests.test_file_public'],public_extra)
assert subprocess.check_output(['git','rev-parse',commit+':src/nanoisa/nvm2c_file_private.c'],env=history_env,text=True).strip()==blob
write('public-history-after.json',{'git_dir':str(history),'commit':commit,'blob':blob,'git':tools()['git']})
# I exclude every archive member's owning object from explicit link inputs.
public=set(shlex.split(values['PUBLIC']))
archive=root/'lib/libnano_file_runtime.a'
linked=[p for p in shlex.split(values['ISA']) if p not in public]
members=subprocess.check_output(['ar','t',str(archive)],text=True).splitlines()
assert not {Path(p).name for p in linked}.intersection(members)
(report/'archive-members.txt').write_text('\n'.join(members)+'\n')
with (report/'archive-nm.txt').open('wb') as output:
 subprocess.run(['nm','-g',str(archive)],stdout=output,stderr=subprocess.STDOUT,check=True)
required={'nvm_file_cyclic_hosted_prepare','nvm_file_execute_bytes','nvm2c_emit_file_bytes','nvm_file_host_grant_create_temporary_files'}
def defined(path):
 found=set()
 for line in path.read_text().splitlines():
  fields=line.split()
  if len(fields)>=3 and fields[-2].upper()!='U':found.add(fields[-1].lstrip('_'))
 return found
assert required <= defined(report/'archive-nm.txt')
for p in linked:
 with (report/('explicit-'+Path(p).name+'-nm.txt')).open('wb') as output:
  subprocess.run(['nm','-g',p],stdout=output,stderr=subprocess.STDOUT,check=True)
 assert not required.intersection(defined(report/('explicit-'+Path(p).name+'-nm.txt')))
write('archive-link-closure.json',{'archive':str(archive),'archive_sha256':sha(archive),'archive_members':members,'excluded_public_objects':sorted(public),'explicit_objects':linked,'member_overlap':[]})
run('archive-boundary',py+['tests.test_file_cyclic_hosted_integration'],dict(FILE_CYCLIC_HOSTED_LDFLAGS=values['LDFLAGS'],FILE_CYCLIC_HOSTED_OBJECTS=shlex.join(linked),FILE_CYCLIC_HOSTED_ARCHIVE=str(archive),NANO_FILE_CYCLIC_HOSTED_CC=os.environ['CC'],NANO_FILE_CYCLIC_HOSTED_CFLAGS=normal_flags))
