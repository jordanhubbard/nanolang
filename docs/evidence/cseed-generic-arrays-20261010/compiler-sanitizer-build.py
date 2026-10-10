from pathlib import Path
import shlex, subprocess
root=Path('/Users/jordanh/Src/nanolang')
work=Path('/private/tmp/nl51-generic-sanitized');work.mkdir(exist_ok=True)
raw=subprocess.check_output(['make','-f','Makefile.gnu','-Bn','nano_virt'],cwd=root,text=True)
lines=raw.replace('\\\n',' ').splitlines()
compiler='/opt/homebrew/opt/llvm/bin/clang'
flags=['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer','-g','-O1']
wanted={'src/parser.c','src/typechecker.c','src/env.c','src/nanovirt/codegen.c'}
replacements={}
link=None
for line in lines:
 try: args=shlex.split(line)
 except ValueError: continue
 if '-o' not in args: continue
 output=args[args.index('-o')+1]
 if output=='bin/nano_virt': link=args
 if '-c' not in args: continue
 source=args[args.index('-c')+1]
 if source not in wanted: continue
 target=work/(Path(source).stem+'.o')
 args[0]=compiler;args[args.index('-o')+1]=str(target)
 subprocess.run(args+flags,cwd=root,check=True)
 replacements[output]=str(target)
assert len(replacements)==4 and link is not None
link=[replacements.get(arg,arg) for arg in link]
link[0]=compiler;link[link.index('-o')+1]=str(work/'nano_virt')
subprocess.run(link+flags,cwd=root,check=True)
print('I instrumented parser, checker, environment and NanoISA codegen, and linked their normal dependencies.')
