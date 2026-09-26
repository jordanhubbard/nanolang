from pathlib import Path
import subprocess,shlex,os
root=Path('/tmp/pr522-union-flow-instrumented');root.mkdir(exist_ok=True)
cc='/opt/homebrew/opt/llvm/bin/clang'
flags=['-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
common=[cc,'-std=c99','-g','-O1','-Wall','-Wextra','-Werror','-Isrc','-Isrc/nanoisa','-D_GNU_SOURCE',*flags]
sources=['ownership_contracts','affine_state','affine_bytecode','verifier']
for source in sources:
 subprocess.run([*common,'-c',f'src/nanoisa/{source}.c','-o',str(root/(source+'.o'))],check=True)
env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1','UBSAN_OPTIONS':'halt_on_error=1'}
for part in ['state','bytecode']:
 recipe=subprocess.check_output(['make','-n','test-affine-'+part],text=True)
 line=next(l for l in recipe.splitlines() if f'-o obj/test_affine_{part} ' in l)
 args=shlex.split(line);args[0]=cc
 args=[str(root/f'test_{part}') if a==f'obj/test_affine_{part}' else str(root/Path(a).name) if a in [f'obj/nanoisa/{source}.o' for source in sources] else a for a in args]
 subprocess.run([*args,'-O1',*flags],check=True)
 subprocess.run([str(root/f'test_{part}')],env=env,check=True)
 macros=[f'-Dmalloc=affine_{part}_test_malloc',f'-Dcalloc=affine_{part}_test_calloc'] if part=='bytecode' else ['-Dmalloc=affine_test_malloc','-Dcalloc=affine_test_calloc']
 if part=='bytecode':macros+=['-Drealloc=affine_bytecode_test_realloc','-DNVM_AFFINE_TEST_VISIT_LIMIT=affine_bytecode_test_visit_limit']
 subprocess.run([*common,*macros,'-c',f'src/nanoisa/affine_{part}.c','-o',str(root/f'{part}_fault.o')],check=True)
 fault=[str(root/f'{part}_fault.o') if a==str(root/f'affine_{part}.o') else str(root/f'test_{part}_fault') if a==str(root/f'test_{part}') else a for a in args]
 subprocess.run([*fault,'-O1',*flags,'-DAFFINE_'+('BYTECODE_' if part=='bytecode' else '')+'ALLOCATION_TEST'],check=True)
 subprocess.run([str(root/f'test_{part}_fault')],env=env,check=True)
print('Instrumented state, bytecode, ownership contracts, verifier and fixtures; remaining dependency objects ordinary. Leak detection enabled, including allocation-injection runs.')
