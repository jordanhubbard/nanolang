from pathlib import Path
import hashlib,json,platform,subprocess,sys

def result(*args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

source=['src/nanolang.h','src/env.c','src/eval.c','src/typechecker.c','tests/test_eval.c','tests/test_eval_global_binding_ownership.c','tests/eval_global_bindings.nano','tests/test_eval_global_bindings.py','tests/nanolang_translator_routes.py','tests/test_imported_globals.py','tests/native_process.py','Makefile.gnu']
source += [str(p) for p in sorted(Path('src_nano/nvm2c').glob('*.nano'))]
tools=['bin/nanoc_c','bin/nano_virt','bin/nano_vm','bin/nvm2c','bin/nano_aot_runtime.o','.test_output/global-bindings-source-compiler']
optional='.test_output/obj-global-bindings-asan/test_eval_global_binding_ownership'
if Path(optional).is_file(): tools.append(optional)
data={'component_commit':result('git','rev-parse','HEAD'),'status':result('git','status','--short'),'platform':platform.platform(),'uname':result('uname','-a'),'cc':result('cc','--version'),'clang':result('clang','--version'),'python':sys.version,'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source},'post_run_tool_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in tools},'limits':['I record hashes after the runs, not a complete before/after SDK closure.','My unittest products are temporary and not archived.','I instrument generated C while using the ordinary AOT runtime.','My C-seed-built development NanoLang driver is not a raw Stage1/Stage2 bootstrap.','My full original LeakSanitizer suite and legacy qualified-global AST-to-C route remain failed.']}
if sys.platform=='darwin': data['sdk']=result('xcrun','--sdk','macosx','--show-sdk-version');data['sdk_path']=result('xcrun','--sdk','macosx','--show-sdk-path');data['os']=result('sw_vers')
Path(sys.argv[1]).write_text(json.dumps(data,indent=2)+'\n')
