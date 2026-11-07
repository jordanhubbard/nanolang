import os, subprocess, json, hashlib
from pathlib import Path
root=Path('/Users/jkh/Src/nanolang-51-harness')
env=os.environ.copy(); env['NANO_SHADOW_TIMEOUT_SECONDS']='300'
drivers=':'.join(str(root/'bin'/name) for name in ('nano_virt','nanoc_stage1','nanoc_stage2'))
for key in ('EFFECTS','PURITY','GENERIC_METADATA','MATCH','AFFINE','AFFINE_USE'): env['NANOLANG_'+key+'_COMPILERS']=drivers
env['NANOLANG_SELFHOST_COMPILER']=str(root/'bin/nanoc_stage2')
env['NANOLANG_OWNED_RECONSTRUCTION_COMPILERS']=' '.join(str(root/'bin'/name) for name in ('nanoc_stage1','nanoc_stage2'))
def run(label,cmd):
    with open('/private/tmp/nanolang-combined-'+label+'.log','w') as log:
        result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    print(label,result.returncode,flush=True)
    if result.returncode: raise SystemExit(result.returncode)
manifest=json.loads((root/'obj/bootstrap-nanoisa/run-cfsrcu5q/manifest.json').read_text())
assert manifest.get('raw_stage1_stage2_equal') is True and manifest['phase']=='stage2' and len(manifest['steps'])==17 and all(s.get('exit_code')==0 for s in manifest['steps']), 'require complete frozen bootstrap'
products=['nano_virt','nanoc_c','nano_vm','nvm2c','nanoisa','nano_aot_runtime.o','nanoc_stage1','nanoc_stage2']
def pins(): return {p:hashlib.sha256((root/'bin'/p).read_bytes()).hexdigest() for p in products}
before=pins(); Path('/private/tmp/nanolang-combined-providers-before.json').write_text(json.dumps(before,indent=2)+'\n')
run('metadata',['make','-f','Makefile.gnu','-o','bootstrap3','-j1','test-checked-borrow-ledger','test-checked-borrow-environment','test-checked-borrow-metadata'])
run('neighbors',['python3','-m','unittest','-v']+['tests.'+s for s in ('test_function_effects_metadata','test_function_purity_metadata','test_generic_binding_metadata','test_match_coverage_metadata','test_function_affine_metadata','test_affine_use_metadata','test_purity_contract')])
run('facts-build',['make','-f','Makefile.gnu','-o','bootstrap3','-j1','nvm2hl'])
run('reconstruction',['python3','-m','unittest','-v']+['tests.'+s for s in ('test_owned_call_pretest_reconstruction','test_owned_shared_control_reconstruction','test_owned_shared_path_reconstruction','test_owned_nested_path_reconstruction')])
after=pins();Path('/private/tmp/nanolang-combined-providers-after.json').write_text(json.dumps(after,indent=2)+'\n');assert before==after,'require frozen provider identity'
