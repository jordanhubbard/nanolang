from pathlib import Path
import shlex, subprocess
root=Path('/private/tmp/nl51-c-layouts')
lines=[x.strip().rstrip('\\').strip().rstrip(';') for x in (root/'build-recipe.txt').read_text().splitlines()]
cc='/opt/homebrew/opt/llvm/bin/clang'
flags=['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
compile_cmd=next(shlex.split(x) for x in lines if x.startswith('cc ') and '-c src/typechecker.c ' in x)
compile_cmd[0]=cc
compile_cmd[compile_cmd.index('-o')+1]=str(root/'typechecker-asan.o')
link_cmd=next(shlex.split(x) for x in lines if x.startswith('cc ') and '-o bin/nanoc_c ' in x)
link_cmd[0]=cc
link_cmd[link_cmd.index('-o')+1]='bin/nanoc_c_layout_asan'
link_cmd=[str(root/'typechecker-asan.o') if x=='obj/typechecker.o' else x for x in link_cmd]
with (root/'sanitizer-build.log').open('w') as log:
    subprocess.run(compile_cmd+flags,check=True,stdout=log,stderr=log)
    subprocess.run(link_cmd+flags,check=True,stdout=log,stderr=log)
