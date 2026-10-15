from pathlib import Path
import subprocess, sys
sys.path.insert(0,str(Path.cwd()))
from tests.test_generic_array_callbacks import GenericArrayCallbacks,PRELUDE
out=Path('/private/tmp/nl51-generic-array-corpus');out.mkdir(exist_ok=True)
for name in sorted(n for n in dir(GenericArrayCallbacks) if n.startswith('test_') and 'incompatible' not in n):
    case=GenericArrayCallbacks(name)
    case.execute=lambda body: (out/(name+'.nano')).write_text(PRELUDE+body+'\nshadow main {assert (== (main) 0)}\n')
    getattr(case,name)()
    source=out/(name+'.nano');module=source.with_suffix('.nvm');c=source.with_suffix('.c');binary=source.with_suffix('')
    commands=[['bin/nano_virt',source,'--emit-nvm','-o',module],['bin/nano_vm',module],['bin/nvm2c',module,'-o',c],['/opt/homebrew/opt/llvm/bin/clang','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',c,'bin/nano_aot_runtime.o','-lm','-o',binary],[binary]]
    for command in commands:
        result=subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=120)
        if result.returncode:
            print(name,'FAILED',command[0],result.stdout,result.stderr,flush=True);break
    else:print(name,'passed C NanoISA, NanoVM and C AOT',flush=True)
