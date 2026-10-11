from pathlib import Path
import json, os, signal, subprocess, sys, tempfile, time, unittest
sys.path.insert(0,str(Path.cwd()))
from tests import test_one_ir_compiler as target
root=Path.cwd()
work=Path(tempfile.mkdtemp(prefix='nl51-dispatch-product-',dir='/private/tmp'))
print('I retain this audit at '+str(work),flush=True)
import importlib.util
spec=importlib.util.spec_from_file_location('bootstrap_inputs',root/'scripts/bootstrap_nanoisa.py')
inputs=importlib.util.module_from_spec(spec);spec.loader.exec_module(inputs)
manifest={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'steps':[],
          'source_inputs':inputs.source_inputs(root),
          'native_test_cc':os.environ.get('NANO_NATIVE_TEST_CC')}
(work/'changes.patch').write_bytes(subprocess.check_output(['git','diff']))
(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
class Retained:
    def __init__(self,*args,**kwargs):self.name=tempfile.mkdtemp(prefix=kwargs.get('prefix','case-'),dir=work)
    def __enter__(self):return self.name
    def __exit__(self,*args):return False
target.tempfile.TemporaryDirectory=Retained
class Audit(target.OneIrCompiler):
    def run_checked(self,args,timeout=180,extra_env=None):
        env=dict(os.environ,NANO_MODULE_PATH=str(root/'modules'));env.update(extra_env or {})
        command=list(map(str,args));label=f'{len(manifest["steps"]):02d}'
        print('I run step '+label+': '+' '.join(command),flush=True)
        start=time.monotonic()
        with (work/(label+'.stdout')).open('wb') as out, (work/(label+'.stderr')).open('wb') as err:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,start_new_session=True)
            try:process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();err.write(b'\nI exceeded the existing product-test deadline.\n')
        stdout=(work/(label+'.stdout')).read_bytes();stderr=(work/(label+'.stderr')).read_bytes()
        manifest['steps'].append({'argv':command,'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-start,'stdout':label+'.stdout','stderr':label+'.stderr'})
        (work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        self.assertEqual(process.returncode,0,(stdout+stderr).decode(errors='replace')[-6000:])
        return stdout
suite=unittest.TestSuite([Audit('test_selfhost_emitted_compiler_to_native_nanoisa_product')])
result=unittest.TextTestRunner(verbosity=2).run(suite)
manifest['final_source_inputs']=inputs.source_inputs(root)
manifest['source_inputs_unchanged']=manifest['source_inputs']==manifest['final_source_inputs']
(work/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
sys.exit(not result.wasSuccessful() or not manifest['source_inputs_unchanged'])
