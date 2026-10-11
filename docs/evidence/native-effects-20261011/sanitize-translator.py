from pathlib import Path
import subprocess,shlex
root=Path('/Users/jordanh/Src/nanolang');work=Path('/private/tmp/nl51-native-effects')
lines=(work/'build-callable.log').read_text().splitlines()
base=shlex.split(next(line for line in lines if ' -c src/nanoisa/nvm2c.c ' in line))
flags=['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
cc='/opt/homebrew/opt/llvm/bin/clang'
for name in ('nvm2c','nvm2c_callables','nvm2c_shape'):
 command=[cc]+[str(work/(name+'-asan.o')) if x=='obj/nanoisa/nvm2c.o' else 'src/nanoisa/'+name+'.c' if x=='src/nanoisa/nvm2c.c' else '-O1' if x=='-O3' else x for x in base[1:]]+flags
 subprocess.run(command,cwd=root,check=True)
 symbols=subprocess.check_output(['nm','-u',str(work/(name+'-asan.o'))],text=True)
 assert '__asan_' in symbols and '__ubsan_' in symbols
link=shlex.split(next(line for line in lines if ' -o bin/nvm2c ' in line and ' -c ' not in line))
replace={f'obj/nanoisa/{name}.o':str(work/(name+'-asan.o')) for name in ('nvm2c','nvm2c_callables','nvm2c_shape')}
replace['bin/nvm2c']=str(work/'nvm2c-asan')
subprocess.run([cc]+[replace.get(x,x) for x in link[1:]]+flags,cwd=root,check=True)
print('I verified instrumentation in translator, callable and shape objects.')
