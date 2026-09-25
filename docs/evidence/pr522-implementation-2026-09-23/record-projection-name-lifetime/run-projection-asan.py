import os,sys,unittest
from pathlib import Path
root=Path('/Users/jordanh/Src/nanolang-pr522-repair');sys.path.insert(0,str(root));os.chdir(root)
os.environ.update(NANO_CC='/opt/homebrew/opt/llvm/bin/clang',NANO_CFLAGS='-O0 -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer',NANO_LDFLAGS='-fsanitize=address,undefined -L/opt/homebrew/opt/openssl@3/lib',ASAN_OPTIONS='detect_leaks=1:detect_stack_use_after_return=1',UBSAN_OPTIONS='halt_on_error=1')
from tests import test_record_array_projection as t
original=t.subprocess.run
def run(command,*args,**kwargs):
 command=list(command)
 if str(command[0])==str(root/'bin/nanoc_c'):command[0]='/tmp/pr522-projection-asan/bin/nanoc_c'
 return original(command,*args,**kwargs)
t.subprocess.run=run
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(t))
sys.exit(not result.wasSuccessful())
