import os
from tests.test_imported_globals import ImportedGlobals
os.environ['NANO_NATIVE_TEST_CC']='/opt/homebrew/opt/llvm/bin/clang'
cases={
'qualified_string_array':('module "a.nano" as first\nfn main()->int{set first.values (array_push first.values "kept") assert (== (at first.values 0) "kept") set first.values [] return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'pub let mut values: array<string> = []\n'}),
'selective_string_array':('from "a.nano" import values\nfn main()->int{set values (array_push values "kept") assert (== (at values 0) "kept") set values [] return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'pub let mut values: array<string> = []\n'}),
'qualified_map':('module "a.nano" as first\nfn main()->int{(map_put first.values "key" "kept") assert (== (map_get first.values "key") "kept") return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'pub let values: HashMap<string,string> = (map_new)\n'}),
'selective_map':('from "a.nano" import values\nfn main()->int{(map_put values "key" "kept") assert (== (map_get values "key") "kept") return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'pub let values: HashMap<string,string> = (map_new)\n'}),
'qualified_record':('module "a.nano" as first\nfn main()->int{assert (== first.value.text "kept") return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'struct Record { text: string }\npub let value: Record = Record { text: "kept" }\n'}),
'selective_record':('from "a.nano" import value\nfn main()->int{assert (== value.text "kept") return 0}\nshadow main {assert (== (main) 0)}',{'a.nano':'struct Record { text: string }\npub let value: Record = Record { text: "kept" }\n'})}
for compiler in ['bin/nano_virt','bin/nano_vm /private/tmp/nanolang-import-callable-corrected-compiler.nvm --']:
 os.environ['NANO_IMPORTED_GLOBAL_COMPILER']=compiler
 for name,(source,files) in cases.items():
  try:
   ImportedGlobals().check(source,files=files)
   print(compiler,name,'PASS',flush=True)
  except Exception as e: print(compiler,name,'FAIL',str(e),flush=True)
