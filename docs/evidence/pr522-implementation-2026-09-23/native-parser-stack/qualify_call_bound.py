import sys,unittest,os
from pathlib import Path
root=Path('/Users/jordanh/Src/nanolang-pr522-repair')
os.chdir(root);sys.path.insert(0,str(root))
from tests import test_one_ir_compiler as suite
if sys.argv[1]=='before':
    original=suite.OneIrCompiler.run_checked
    def baseline(self,args,*pos,**kw):
        if str(args[0])==str(root/'bin/nvm2c'):
            args=[Path('/tmp/pr522-native-parser-stack/nvm2c-before'),*args[1:]]
        return original(self,args,*pos,**kw)
    suite.OneIrCompiler.run_checked=baseline
if 'instrumented' in sys.argv:
    os.environ.update(NANO_NATIVE_TEST_CC='/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined -fno-sanitize-recover=all',ASAN_OPTIONS='detect_leaks=1:detect_stack_use_after_return=1',UBSAN_OPTIONS='halt_on_error=1')
tests=unittest.defaultTestLoader.loadTestsFromName('OneIrCompiler.test_many_record_calls_fit_bounded_stack',suite)
result=unittest.TextTestRunner(verbosity=2).run(tests)
sys.exit(not result.wasSuccessful())
