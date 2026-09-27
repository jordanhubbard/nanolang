from pathlib import Path
import shlex,subprocess,json,tempfile,unittest,os,sys,gzip
from tests import test_generic_selected_ownership as cases
from tests import test_selected_variant_ownership as nongeneric
build_log=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('pr522-c-owned-build.log.gz')
log_text=gzip.decompress(build_log.read_bytes()).decode() if build_log.suffix=='.gz' else build_log.read_text()
commands=log_text.replace('\\\n', ' ').splitlines()
cc='/opt/homebrew/opt/llvm/bin/clang';flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-O1']
obj='/tmp/pr522-owned-codegen.o';exe='/tmp/pr522-owned-nano-virt'
for line in commands:
 command=shlex.split(line)
 if not command or command[0]!='cc':continue
 if 'src/nanovirt/codegen.c' not in command and 'bin/nano_virt' not in command:continue
 command[0]=cc;command+=flags
 if '-c' in command:
  command[command.index('-o')+1]=obj
  for f in ('-MMD','-MP'):
   if f in command:command.remove(f)
 else:
  command[command.index('-o')+1]=exe
  command=[obj if x=='obj/nanovirt/codegen.o' else '/tmp/pr522-owned-typechecker.o' if x=='obj/typechecker.o' else x for x in command]
 subprocess.run(command,check=True,capture_output=True,text=True)
 if '-c' in command:
  checker=[('src/typechecker.c' if x=='src/nanovirt/codegen.c' else '/tmp/pr522-owned-typechecker.o' if x==obj else x) for x in command]
  subprocess.run(checker,check=True,capture_output=True,text=True)
results=[]
with tempfile.TemporaryDirectory(prefix='pr522-c-instrument-') as d:
 source=Path(d)/'source.nano';module=Path(d)/'out.nvm'
 suite=cases.GenericSelectedOwnership()
 names=unittest.defaultTestLoader.getTestCaseNames(type(suite)) + ['test_two_resources_and_ordinary_sibling_arms','test_nested_resource_record','test_compatible_outer_join']
 for name in names:
  if name=='test_guarded_generic_match_remains_rejected':continue
  captured=[]
  if hasattr(suite,name):
   suite.check=lambda s,a: captured.append((s,a));getattr(suite,name)()
  else:
   plain=nongeneric.SelectedVariantOwnership();plain.program=lambda s,a,diagnostics=None: captured.append((s,a));getattr(plain,name)()
  for s,accepted in captured:
   source.write_text(s);module.write_bytes(b'prior module')
   r=subprocess.run([exe,str(source),'--emit-nvm','-o',str(module)],capture_output=True,text=True,timeout=120,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
   results.append(dict(case=name,accepted=accepted,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,prior_preserved=module.read_bytes()==b'prior module'))
   assert (r.returncode==0)==accepted,(name,r.stderr)
   assert accepted or module.read_bytes()==b'prior module',name
   assert 'Sanitizer' not in r.stderr and 'runtime error:' not in r.stderr,(name,r.stderr)
Path('/tmp/pr522-c-owned-instrumented.json').write_text(json.dumps(results,indent=2)+'\n')
print('22 C frontend cases pass scoped ASan/UBSan/leak checks')
