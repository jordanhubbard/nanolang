from pathlib import Path
import subprocess,shlex,os
from tests.test_native_variant_scalar_carriers import VariantScalarCarriers
root=Path('/tmp/pr522-variant-instrumented');root.mkdir(exist_ok=True)
cc='/opt/homebrew/opt/llvm/bin/clang';flags=['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
subprocess.run([cc,'-std=c99','-g','-O1','-Isrc','-Isrc/nanoisa','-D_GNU_SOURCE',*flags,'-c','src/nanoisa/nvm2c.c','-o',str(root/'nvm2c.o')],check=True)
recipe=subprocess.check_output(['make','-n','-W','src/nanoisa/nvm2c.c','nvm2c'],text=True)
line=next(l for l in recipe.splitlines() if '-o bin/nvm2c ' in l)
args=shlex.split(line);args[0]=cc
args=[str(root/'nvm2c') if a=='bin/nvm2c' else str(root/'nvm2c.o') if a=='obj/nanoisa/nvm2c.o' else a for a in args]
subprocess.run([*args,*flags],check=True)
env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1','UBSAN_OPTIONS':'halt_on_error=1'}
case=VariantScalarCarriers()
cases=[(case.guarded_return(),0),(case.guarded_return(tag=0),1),(case.guarded_return(extra='FUNCREF read\nPOP\n'),1),(case.guarded_return(bypass=True),1)]
for index,(text,expected) in enumerate(cases):
 assembly=root/f'{index}.nasm';module=root/f'{index}.nvm';out=root/f'{index}.c'
 assembly.write_text(text)
 subprocess.run(['bin/nanoisa','asm',str(assembly),'-o',str(module)],check=True)
 out.write_text('prior output')
 r=subprocess.run([str(root/'nvm2c'),str(module),'-o',str(out)],env=env,capture_output=True,text=True)
 print('case',index,'exit',r.returncode,'expected',expected,r.stdout,r.stderr,flush=True)
 assert r.returncode==expected
 if expected:assert out.read_text()=='prior output'
print('Instrumented translation unit: admission, refusal and cleanup passed.')
