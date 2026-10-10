from pathlib import Path
import os,subprocess,unittest
from unittest.mock import patch
from tests.test_bootstrap_components import BootstrapComponents
root=Path.cwd()
work=root/'coverage-env-probe';work.mkdir(exist_ok=True)
original=subprocess.run
objects=[]
for name in ('dyn_array','gc','gc_struct'):
 obj=work/(name+'.o');objects.append(obj)
 subprocess.run(['cc','-std=c99','-D_GNU_SOURCE','-Isrc','--coverage','-c',str(root/'src/runtime'/ (name+'.c')),'-o',str(obj)],check=True)
runtime=work/'runtime.o'
subprocess.run(['cc','-r','-nostdlib',*map(str,objects),'-o',str(runtime)],check=True)
print('LDFLAGS:',repr(os.environ.get('LDFLAGS')),flush=True)
# I execute the real harness, substituting only its isolated instrumented runtime.
def run(args,*pos,**kwargs):
 args=[str(runtime) if str(a)==str(root/'bin/nano_aot_runtime.o') else a for a in args]
 return original(args,*pos,**kwargs)
with patch('subprocess.run',run):
 result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([
  BootstrapComponents('test_real_emitter_driver_executes_in_vm_and_sanitized_native')]))
raise SystemExit(not result.wasSuccessful())
