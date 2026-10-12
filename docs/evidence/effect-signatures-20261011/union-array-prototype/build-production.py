from pathlib import Path
import subprocess,shlex
w=Path('/private/tmp/nl51-union-arrays')
commands=subprocess.check_output(['make','-n','bin/nvm2c'],text=True)
args=shlex.split(next(x for x in commands.splitlines() if '-o bin/nvm2c ' in x))
args[args.index('bin/nvm2c')]=str(w/'nvm2c');args[args.index('obj/nanoisa/nvm2c.o')]=str(w/'nvm2c.o')
with (w/'production-build.log').open('w') as log:
 subprocess.run(['/opt/homebrew/opt/llvm/bin/clang','-std=c99','-Wall','-Wextra','-Werror','-D_GNU_SOURCE','-Isrc','-Isrc/nanoisa','-O0','-g','-c','src/nanoisa/nvm2c.c','-o',str(w/'nvm2c.o')],stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
