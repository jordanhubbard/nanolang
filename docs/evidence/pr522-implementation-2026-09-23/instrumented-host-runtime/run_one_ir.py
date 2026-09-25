import os,sys,unittest
from pathlib import Path
root=Path('/Users/jordanh/Src/nanolang-pr522-repair')
os.chdir(root)
sys.path.insert(0,str(root))
os.environ.update(NANO_NATIVE_TEST_CC='/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer',NANO_CC='/opt/homebrew/opt/llvm/bin/clang',NANO_LDFLAGS='-fsanitize=address,undefined -L/opt/homebrew/opt/openssl@3/lib',NANO_SHADOW_TIMEOUT_SECONDS='60',ASAN_OPTIONS='detect_leaks=1:detect_stack_use_after_return=1',UBSAN_OPTIONS='halt_on_error=1')
os.environ.pop('NANO_ARTIFACT_LDFLAGS',None)
from tests import test_one_ir_compiler as suite
suite.HOST_RUNTIME[0]=Path('/tmp/pr522-final-qualification/nano_aot_runtime.o')
print('I run every OneIrCompiler method with a fresh ASan/UBSan HOST_RUNTIME and instrumented generated native products.',flush=True)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(suite))
sys.exit(not result.wasSuccessful())
