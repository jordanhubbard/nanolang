from pathlib import Path
import subprocess,shlex,os
root=Path('/tmp/pr522-owned-transport-instrumented');root.mkdir(exist_ok=True)
cc='/opt/homebrew/opt/llvm/bin/clang'
flags=['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
for source in ['ownership_contracts','affine_state','verifier']:
 subprocess.run([cc,'-std=c99','-g','-O1','-Wall','-Wextra','-Werror','-Isrc','-Isrc/nanoisa','-D_GNU_SOURCE',*flags,'-c',f'src/nanoisa/{source}.c','-o',str(root/(source+'.o'))],check=True)
recipe=subprocess.check_output(['make','-n','test-ownership-contracts'],text=True)
line=next(l for l in recipe.splitlines() if '-o obj/test_ownership_contracts ' in l)
args=shlex.split(line);args[0]=cc
args=[str(root/'test') if a=='obj/test_ownership_contracts' else str(root/(Path(a).name)) if a in ['obj/nanoisa/ownership_contracts.o','obj/nanoisa/affine_state.o','obj/nanoisa/verifier.o'] else a for a in args]
subprocess.run([*args,*flags],check=True)
env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1','UBSAN_OPTIONS':'halt_on_error=1'}
subprocess.run([str(root/'test')],env=env,check=True)
print('Instrumented ownership contracts, affine state, verifier and test fixture; other dependency objects are ordinary builds.')
