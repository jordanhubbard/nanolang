from pathlib import Path
import shlex,subprocess,os,tempfile,sys
root=Path.cwd();cc='/opt/homebrew/opt/llvm/bin/clang'
commands=subprocess.check_output(['make','-n','test-owned-union-runtime'],text=True).splitlines()
recipe=next(shlex.split(x) for x in commands if ' -o obj/test_owned_union_runtime ' in x)
flags=recipe[1:recipe.index('-o')];san=['-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer']
if '--switch' in sys.argv:san.append('-DNANO_NO_COMPUTED_GOTO')
with tempfile.TemporaryDirectory(prefix='pr522-runtime-instrument-') as d:
 work=Path(d); replacements={}
 for source in ['src/nanovm/vm.c','src/nanovm/heap.c','src/nanovm/heap_cycles.c','src/nanoisa/nvm2c.c']:
  obj=work/(Path(source).stem+'.o');subprocess.run([cc,*flags,*san,'-c',source,'-o',str(obj)],check=True)
  replacements['obj/'+source[4:-2]+'.o']=str(obj)
 env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'}
 for line in commands:
  if not line.startswith('cc ') or ' -o obj/test_owned_union' not in line:continue
  args=shlex.split(line);args[0]=cc;args=[replacements.get(x,x) for x in args]
  out=args.index('-o')+1;original=args[out];args[out]=str(work/Path(original).name);args+=san
  subprocess.run(args,check=True)
  if original.endswith('heap_alloc.o'):
   replacements[original]=args[out];continue
  target=Path(args[out]);run=[str(target)]
  if target.name=='test_owned_union_runtime':
   artifacts=work/'artifacts';artifacts.mkdir();run.append(str(artifacts))
  subprocess.run(run,check=True,env=env)
 print('Scoped VM, heap, cycle collector, native emitter and fixture ASan/UBSan/LSAN1 checks passed')
