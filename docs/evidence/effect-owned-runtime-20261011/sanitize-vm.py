from pathlib import Path
import shlex, subprocess, os
root=Path('/Users/jordanh/Src/nanolang')
work=Path('/private/tmp/nl51-native-effects')
lines=Path('/private/tmp/nl51-native-effects-runtime-expanded.log').read_text().splitlines()
outputs=['obj/test_affine_effect_vm_alloc.o','obj/test_affine_effect_heap_alloc.o','obj/test_affine_effect_vm']
for output in outputs:
    line=next(x for x in lines if x.startswith('cc ') and f'-o {output}' in x and shlex.split(x)[shlex.split(x).index('-o')+1]==output)
    args=shlex.split(line)
    args[0]='/opt/homebrew/opt/llvm/bin/clang'
    args=[str(work/Path(x).name) if x in outputs else x for x in args]
    args += ['-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
    print(shlex.join(args),flush=True)
    subprocess.run(args,cwd=root,check=True)
env={**os.environ,'NANO_AFFINE_EFFECT_VM':str(work/'test_affine_effect_vm'),
     'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'}
subprocess.run(['python3','-m','unittest','-v','tests.test_affine_effect_runtime'],cwd=root,env=env,check=True)
