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
 paths={n:shutil.which(n) for n in ('make','python3','ar','ld','cc','gcc','clang','pkg-config')}
 paths['driver_python']=sys.executable
 if darwin:
  paths['xcrun']='/usr/bin/xcrun'
  paths['apple_clang']=subprocess.check_output(['xcrun','--find','clang'],text=True).strip()
  paths['apple_ld']=subprocess.check_output(['xcrun','--find','ld'],text=True).strip()
 paths.update(ordinary=shutil.which(os.environ['CC']),sanitizer=shutil.which(os.environ['CARRIER_SAN_CC']))
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
  if not dest.exists():
   shared=None
   for prior in ('3115c','4dc1c','88ae0'):
    candidate=Path('/tmp/nanolang-record-vm-'+prior+'-'+('puck' if darwin else 'linux'))/'artifacts'/digest
    if candidate.is_file() and sha(candidate)==digest:shared=candidate;break
   if shared is not None:os.link(shared,dest)
   else:shutil.copyfile(p,dest)
  entries[str(p)]={'sha256':digest,'artifact':str(dest)}
 return entries
write('environment.json',{k:v for k,v in os.environ.items() if k in ('CC','CARRIER_SAN_CC','SDKROOT','LIBRARY_PATH','NMS_NATIVE_CLANG_FLAGS','PATH','CARRIER_PHASES','NANO_FILE_RUNTIME_CFLAGS','CARRIER_EXTRA_TOOLS','LSAN_OPTIONS','ASAN_OPTIONS','UBSAN_OPTIONS')})
for label,compiler in [('ordinary',os.environ['CC']),('sanitizer',os.environ['CARRIER_SAN_CC']),('clang',shutil.which('clang'))]:
 if compiler:
  p=subprocess.run([compiler,'--version'],capture_output=True,text=True);(report/(label+'-version.txt')).write_text(p.stdout+p.stderr)
if darwin:
 p=subprocess.run(['xcrun','--show-sdk-version'],capture_output=True,text=True);(report/'sdk-version.txt').write_text(p.stdout+p.stderr)
shutil.copyfile(__file__,report/'driver.py');results=[]
def run(label,cmd,extra=None):
 if phases and label not in phases:return
 free=shutil.disk_usage(report).free;write(label+'-space-before.json',{'free_bytes':free,'minimum_bytes':1073741824})
 if free<1073741824:write(label+'-terminal.json',{'launched':False,'reason':'capacity preflight','free_bytes':free});raise SystemExit(125)
 before=source();toolbefore=tools();write(label+'-source-before.json',before);write(label+'-tools-before.json',toolbefore);write(label+'-inputs-before.json',inventory())
 env=dict(os.environ);env.update(extra or {})
 write(label+'-command.json',{'argv':cmd,'selectors':{k:v for k,v in env.items() if k.startswith(('NANO_','FILE_','WRAPPER_','ORDINARY_','DECLARATION_','RECORD_ARRAY_')) or k in ('CC','SDKROOT','LIBRARY_PATH','NMS_NATIVE_CLANG_FLAGS','LSAN_OPTIONS')}})
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
 text=(report/(label+'.log')).read_text(errors='replace');dirs=re.findall(r'I retain [^\n]*? artifacts at (\S+)',text)
 # Existing neighbors occasionally use a differently worded retention message.
 dirs+=re.findall(r'(/(?:tmp|var/folders)/[^\s\'\"]*nano-[^\s\'\"]+)',text)
 write(label+'-artifacts.json',inventory([*dirs,report/'controls']));print(json.dumps(result),flush=True)
 assert before==after and toolbefore==toolafter,'source or host tools changed during phase'
 if completed.returncode:sys.exit(completed.returncode)

os.environ['LSAN_OPTIONS']=''
controls=report/'controls';controls.mkdir()
probe=report/'providers.mk'
probe.write_text("\n.PHONY: array-providers array-config\narray-providers: $(NANOISA_OBJECTS) $(NANOISA_UTF8)\narray-config:\n\t@printf '%s\\n' 'ISA=$(NANOISA_OBJECTS) $(NANOISA_UTF8)' 'LDFLAGS=$(LDFLAGS)'\n")
run('setup',['make','-f','Makefile.gnu','-f',str(probe),'-j2','CC='+os.environ['CC'],'array-providers','nano_vm','nvm2c'])
run('configuration',['make','-s','-f','Makefile.gnu','-f',str(probe),'CC='+os.environ['CC'],'array-config'])
values=dict(line.split('=',1) for line in (report/'configuration.log').read_text().splitlines() if '=' in line)
objects=shlex.split(values['ISA']);links=shlex.split(values['LDFLAGS'])
filtered=' '.join(p for p in objects if Path(p).stem not in ('ownership_contracts','nvm_v2_layouts'))
base=dict(RECORD_ARRAY_OBJECTS=' '.join(objects),RECORD_ARRAY_LDFLAGS=values['LDFLAGS'])
py=[sys.executable,'-m','unittest','-f','-v','tests.test_record_array_global_flow']
clang=shutil.which('clang');clang_flags='' if darwin else '--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'
sanitize=' -fsanitize=address,undefined -fno-omit-frame-pointer'
run('discovery',[sys.executable,'-c',"import unittest;from tests import test_record_array_global_flow as m;s=unittest.defaultTestLoader.loadTestsFromModule(m);print(s.countTestCases());assert s.countTestCases()==1"])
run('ordinary',py,dict(base,RECORD_ARRAY_CC=os.environ['CC'],RECORD_ARRAY_CFLAGS=''))
run('clang-ordinary',py,dict(base,RECORD_ARRAY_CC=clang,RECORD_ARRAY_CFLAGS=clang_flags))
run('sanitizer',py,dict(base,RECORD_ARRAY_CC=os.environ['CARRIER_SAN_CC'],RECORD_ARRAY_CFLAGS=sanitize))
if not darwin:run('clang-sanitizer',py,dict(base,RECORD_ARRAY_CC=clang,RECORD_ARRAY_CFLAGS=clang_flags+sanitize))
for name in ('ownership_contracts','ordinary_record_authority','nvm_v2_layouts','ownership_layouts_private','nvm_v2_convert','nvm_v2_endtoend'):
 selected=[p for p in objects if name!='ordinary_record_authority' or Path(p).stem not in ('ownership_contracts','nvm_v2_layouts')]
 exe=controls/name
 run(name+'-build',[os.environ['CC'],'-std=c11','-D_DEFAULT_SOURCE','-O1','-g','-Wall','-Wextra','-Werror','-Isrc/nanoisa','-Imodules/nanoisa','tests/nanoisa/test_'+name+'.c',*selected,*links,'-o',str(exe)])
 run(name+'-run',[str(exe)])

run('old-array',[sys.executable,'-m','unittest','-f','-v','tests.test_ordinary_array_authority'],dict(ORDINARY_ARRAY_OBJECTS=filtered,ORDINARY_ARRAY_LDFLAGS=values['LDFLAGS'],ORDINARY_ARRAY_CC=os.environ['CC'],ORDINARY_ARRAY_CFLAGS=''))
run('affine-union',['make','-j2','CC='+os.environ['CC'],'test-affine-scalar-union-runtime'],dict(NANO_AFFINE_UNION_CC=os.environ['CARRIER_SAN_CC']))
run('ownership-public',['make','-j2','CC='+os.environ['CC'],'test-ownership-contracts'])

run('complete-declarations',[sys.executable,'-m','unittest','-f','-v','tests.test_ownership_declaration_projection'],dict(DECLARATION_OBJECTS=filtered,DECLARATION_LDFLAGS=values['LDFLAGS'],DECLARATION_CC=os.environ['CC'],DECLARATION_CFLAGS=''))

for name in ('record_array_origins','record_array_execution'):
 run(name,[sys.executable,'-m','unittest','-f','-v','tests.test_'+name],dict(base,RECORD_ARRAY_CC=os.environ['CC'],RECORD_ARRAY_CFLAGS=''))
