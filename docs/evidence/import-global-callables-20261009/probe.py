import os
from tests.test_imported_globals import ImportedGlobals
os.environ['NANO_NATIVE_TEST_CC']='/opt/homebrew/opt/llvm/bin/clang'
cases={
'array_qualified':('module "a.nano" as first\nfn main()->int{assert (== (at first.values 1) 7) return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'pub let values: array<int> = [3,7]\n'}),
'array_selective':('from "a.nano" import values\nfn main()->int{assert (== (at values 1) 7) return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'pub let values: array<int> = [3,7]\n'}),
'callable_qualified':('module "a.nano" as first\nfn main()->int{assert (== (first.apply 4) 5) return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'pub let apply: fn(int)->int = fn(x:int)->int{return (+ x 1)}\n'}),
'callable_selective':('from "a.nano" import apply\nfn main()->int{assert (== (apply 4) 5) return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'pub let apply: fn(int)->int = fn(x:int)->int{return (+ x 1)}\n'}),
'diamond_init':('module "left.nano" as left\nmodule "right.nano" as right\nmodule "a.nano" as first\nfn main()->int{assert (== (left.read) 1) assert (== (right.read) 1) assert (== first.count 1) return 0}\nshadow main {assert (== (main) 0)}',{
'a.nano':'pub let mut count: int = 0\nfn next()->int{set count (+ count 1) return count}\nshadow next {let saved: int = count set count 0 assert (== (next) 1) set count saved}\npub let value: int = (next)\n',
'left.nano':'module "a.nano" as first\npub fn read()->int{return first.value}\nshadow read {assert (== (read) 1)}\n',
'right.nano':'module "./a.nano" as first\npub fn read()->int{return first.value}\nshadow read {assert (== (read) 1)}\n'})}
for compiler in ['bin/nano_virt','bin/nano_vm /private/tmp/nanolang-qualified-write-corrected-compiler.nvm --']:
 os.environ['NANO_IMPORTED_GLOBAL_COMPILER']=compiler
 for name,(source,files) in cases.items():
  try:
   ImportedGlobals().check(source,files=files)
   print(compiler,name,'PASS',flush=True)
  except Exception as e: print(compiler,name,'FAIL',str(e),flush=True)
