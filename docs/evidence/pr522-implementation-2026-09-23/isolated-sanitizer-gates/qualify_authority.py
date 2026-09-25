from pathlib import Path
import json,subprocess,os,shlex,sys
mode=sys.argv[1]
base=Path('/tmp/pr522-isolated-gates')
objs=json.loads((base/'objects.json').read_text())[mode]
env=dict(os.environ,NOA_LINK_OBJECTS=shlex.join(objs),NANOLANG_GUARD_SAN_CC='/opt/homebrew/opt/llvm/bin/clang',NANO_LLC='/opt/homebrew/opt/llvm/bin/llc',NANO_WASM_LD='/opt/homebrew/opt/lld/bin/wasm-ld',ASAN_OPTIONS='detect_leaks=1:detect_stack_use_after_return=1',UBSAN_OPTIONS='halt_on_error=1')
for key in ('NANO_NATIVE_TEST_CC','NANO_CC','CC','NANO_ARTIFACT_LDFLAGS'):
    env.pop(key,None)
env['NANO_LDFLAGS']='-L/opt/homebrew/opt/openssl@3/lib'
if mode=='instrumented':
    env['NANO_CC']='/opt/homebrew/opt/llvm/bin/clang'
    env['NANO_LDFLAGS']+=' -fsanitize=address,undefined'
with (base/f'authority-{mode}-final.log').open('w') as log:
    result=subprocess.run(['python3','-m','unittest','tests.test_ordinary_record_authority','-v'],env=env,stdout=log,stderr=subprocess.STDOUT)
print(mode,'exit',result.returncode)
raise SystemExit(result.returncode)
