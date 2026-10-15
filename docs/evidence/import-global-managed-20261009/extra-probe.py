import os,subprocess
from tests.test_imported_globals import ImportedGlobals
os.environ['NANO_NATIVE_TEST_CC']='/opt/homebrew/opt/llvm/bin/clang'
real_run=subprocess.run
def run(command,*args,**kwargs):
 command=list(command)
 if str(command[0]).endswith('/bin/nvm2c'):command[0]='/private/tmp/nanolang-record-nvm2c'
 return real_run(command,*args,**kwargs)
subprocess.run=run
cases={
'record_module_identity':('module "a.nano" as first\nmodule "b.nano" as second\nfn main()->int{assert (== first.box.value 41) assert (== second.box.value "second") return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'struct Box { value: int }\npub let box: Box = Box { value: 41 }\n','b.nano':'struct Box { value: string }\npub let box: Box = Box { value: "second" }\n'}),
'closure_global_capture':('module "a.nano" as first\nfn main()->int{assert (== (first.apply 4) 11) return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'fn make(base:int)->fn(int)->int{return fn(value:int)->int{return (+ base value)}}\nshadow make {let f:fn(int)->int=(make 3) assert (== (f 4) 7)}\npub let apply:fn(int)->int=(make 7)\n'}),
'call_snapshot_before_argument':('module "a.nano" as first\nfn main()->int{let result:int=(first.apply (first.swap)) assert (== result 2) assert (== (first.apply 1) 101) (first.reset) return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'pub let mut apply:fn(int)->int=fn(x:int)->int{return (+ x 1)}\npub fn swap()->int{set apply fn(x:int)->int{return (+ x 100)} return 1}\nshadow swap {let saved:fn(int)->int=apply assert (== (swap) 1) assert (== (apply 1) 101) set apply saved}\npub fn reset()->void{set apply fn(x:int)->int{return (+ x 1)}}\nshadow reset {(reset) assert (== (apply 1) 2)}\n'})}
for compiler in ['/private/tmp/nanolang-map-nano-virt','bin/nano_vm /private/tmp/nanolang-import-callable-corrected-compiler.nvm --']:
 os.environ['NANO_IMPORTED_GLOBAL_COMPILER']=compiler
 for name,(source,files) in cases.items():
  try:
   ImportedGlobals().check(source,files=files)
   print(compiler,name,'PASS',flush=True)
  except Exception as e:print(compiler,name,'FAIL',str(e),flush=True)
