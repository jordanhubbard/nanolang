import hashlib,json,os,re,shlex,shutil,subprocess,sys,time
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
write('environment.json',{k:v for k,v in os.environ.items() if k in ('CC','CARRIER_SAN_CC','SDKROOT','LIBRARY_PATH','NMS_NATIVE_CLANG_FLAGS','PATH','CARRIER_PHASES','NANO_FILE_RUNTIME_CFLAGS','CARRIER_EXTRA_TOOLS','LSAN_OPTIONS','ASAN_OPTIONS','UBSAN_OPTIONS')})
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
 start=time.monotonic()
 with (report/(label+'.log')).open('w') as out:completed=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
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
config=subprocess.check_output(['make','-s','-f','Makefile.gnu','-f',str(probe),'carrier-config'],text=True)
(report/'provider-config.txt').write_text(config);values=dict(line.split('=',1) for line in config.splitlines() if '=' in line)
base={'FILE_RUNTIME_OBJECTS':values['ALL'],'FILE_RUNTIME_LDFLAGS':values['LDFLAGS']}
py=[sys.executable,'-m','unittest','-f','-v']
run('normal',py+['tests.test_file_private_vm'],dict(base,NANO_FILE_RUNTIME_CC=os.environ['CC'],NANO_FILE_RUNTIME_SANITIZERS='0'))
run('sanitizer',py+[os.environ.get('CARRIER_RUNTIME_SAN_TEST','tests.test_file_private_vm')],dict(base,NANO_FILE_RUNTIME_CC=os.environ['CARRIER_SAN_CC'],NANO_FILE_RUNTIME_SANITIZERS='1'))
if not darwin:run('clang',py+['tests.test_file_private_vm'],dict(base,NANO_FILE_RUNTIME_CC=shutil.which('clang'),NANO_FILE_RUNTIME_CFLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',NANO_FILE_RUNTIME_SANITIZERS='1'))
run('frames',py+['tests.test_file_runtime_frames'],dict(base,NANO_FILE_RUNTIME_CC=os.environ['CARRIER_SAN_CC'],NANO_FILE_RUNTIME_SANITIZERS='1'))
compiler=os.environ['CARRIER_SAN_CC'];common={'NANO_NATIVE_TEST_CC':compiler,'NANO_FILE_VALUES_CC':compiler,'NANO_FILE_OPCODE_TEST_CC':compiler}
run('hosted',py+['tests.test_file_hosted'],dict(common,FILE_HOSTED_OBJECTS=values['ISA'],FILE_HOSTED_LDFLAGS=values['LDFLAGS'],NANO_FILE_HOSTED_CC=compiler))
filtered=' '.join(p for p in shlex.split(values['ISA']) if Path(p).stem not in ('file_flow','service_file_nominal','service_file_nominal_plan','nsi_file_plan'))
for name in ('body','flow'):
 opts=dict(common);opts.update({f'FILE_{name.upper()}_OBJECTS':filtered,f'FILE_{name.upper()}_LDFLAGS':values['LDFLAGS'],f'NANO_FILE_{name.upper()}_CC':compiler})
 run(name,py+[f'tests.test_file_{name}'],opts)
run('values',py+['tests.test_nsi_file_values'],common)
run('opcodes',py+['tests.test_file_opcodes'],dict(common,FILE_OPCODE_OBJECTS=values['OPCODES'],FILE_OPCODE_LDFLAGS=values['LDFLAGS']))
retention="""import tempfile,unittest
class Retained(tempfile.TemporaryDirectory):
 def cleanup(self):
  if self._finalizer.detach():print('I retain wrapper artifacts at '+self.name,flush=True)
tempfile.TemporaryDirectory=Retained
unittest.main(module=None,argv=['unittest','-f','-v','tests.test_wrapper_publication'])
"""
run('wrapper',[sys.executable,'-c',retention],common)
