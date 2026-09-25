import os,sys,unittest
from pathlib import Path
root=Path('/Users/jordanh/Src/nanolang-pr522-repair')
os.chdir(root)
sys.path.insert(0,str(root))
os.environ.update(NANO_NATIVE_TEST_CC='/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer',NANO_CC='/opt/homebrew/opt/llvm/bin/clang',NANO_LDFLAGS='-fsanitize=address,undefined -L/opt/homebrew/opt/openssl@3/lib',NANO_SHADOW_TIMEOUT_SECONDS='60',ASAN_OPTIONS='detect_leaks=1:detect_stack_use_after_return=1',UBSAN_OPTIONS='halt_on_error=1')
os.environ.pop('NANO_ARTIFACT_LDFLAGS',None)
from tests import test_one_ir_compiler as suite
suite.HOST_RUNTIME[0]=Path('/tmp/pr522-final-qualification/nano_aot_runtime.o')
original_run = suite.OneIrCompiler.run_checked
def prototype_run(self,args,*pos,**kw):
    result=original_run(self,args,*pos,**kw)
    if str(args[0]).endswith('/bin/nvm2c') and '-o' in args:
        output=Path(args[args.index('-o')+1])
        text=output.read_text()
        header=(root/'src/runtime/dyn_array.h').read_text()
        decl=header[header.index('typedef enum {'):header.index('} DynArray;')+len('} DynArray;')]
        start=text.index('typedef enum { nh_int=1')
        end=text.index('} nh_array_value;',start)+len('} nh_array_value;')
        text=text[:start]+decl+text[end:]
        text=text.replace('nh_array_value','DynArray').replace('foreign->type','foreign->elem_type').replace('foreign->width','foreign->elem_size').replace('nh_string','ELEM_STRING')
        output.write_text(text)
    return result
suite.OneIrCompiler.run_checked=prototype_run
print('I diagnose the real std artifact with an exact-declaration adapter prototype.',flush=True)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('OneIrCompiler.test_real_std_artifact_uses_host_runtime',suite))
sys.exit(not result.wasSuccessful())
