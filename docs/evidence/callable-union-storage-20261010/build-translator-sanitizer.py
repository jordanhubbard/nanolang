from pathlib import Path
import subprocess,shlex
root=Path('/Users/jordanh/Src/nanolang');work=Path('/private/tmp/nl51-context-storage/sanitizer');work.mkdir(exist_ok=True)
compiler='/opt/homebrew/opt/llvm/bin/clang';flags=['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer','-O1','-g']
raw=subprocess.check_output(['make','-f','Makefile.gnu','-Bn','bin/nvm2c'],cwd=root,text=True).replace('\\\n',' ')
wanted={'src/nanoisa/nvm2c.c','src/nanoisa/nvm2c_shape.c','src/nanoisa/nvm2c_callables.c'};replacements={};link=None
for line in raw.splitlines():
 try:args=shlex.split(line)
 except ValueError:continue
 if not args or args[0]!='cc' or '-o' not in args:continue
 output=args[args.index('-o')+1]
 if '-c' in args and args[args.index('-c')+1] in wanted:
  source=args[args.index('-c')+1];target=work/(Path(source).stem+'.o')
  args[0]=compiler;args[args.index('-o')+1]=str(target)
  subprocess.run(args+flags,cwd=root,check=True);replacements[output]=str(target)
 if output=='bin/nvm2c':link=args
assert link and len(replacements)==3
link=[replacements.get(arg,arg) for arg in link];link[0]=compiler;link[link.index('-o')+1]=str(work/'nvm2c')
subprocess.run(link+flags,cwd=root,check=True)
for target in replacements.values():
 symbols=subprocess.check_output(['nm','-u',target],text=True)
 assert '__asan_' in symbols and '__ubsan_' in symbols
link=[('tests/nanoisa/test_nvm2c_callables.c' if x=='obj/nanoisa/nvm2c_main.o' else x) for x in link]
link[link.index('-o')+1]=str(work/'callables')
subprocess.run(link+flags+['-Isrc/nanoisa'],cwd=root,check=True)
print('I verified ASan/UBSan instrumentation in all three translator objects.')
