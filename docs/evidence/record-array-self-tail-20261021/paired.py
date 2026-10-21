exec(open('/private/tmp/nanolang-self-tail-proof-20261021/run.py').read().split("run('changed-object'")[0])
import os,unittest
from tests import test_flat_array_source_publication as source_tests
cache=work/'module-cache';query_objects=sorted(cache.glob('*/.nano-gen-*/nanoisa_*.o'))
assert query_objects
library=work/'libnanoisa_self_tail.so'
run('query-library',[cc,'-dynamiclib',*query_objects,root/'obj/utf8.o',*ld,'-o',library])
class CurrentPublication(source_tests.FlatArraySourcePublication):
 @classmethod
 def setUpClass(cls):
  cls.compilers=[str(child/'bin/nano_virt_self_tail'),str(work/'emitter')]
  cls.query_library=library
 def run_command(self,*args,**kw):
  mapping={str(child/'bin'/name):str(child/'bin'/(name+'_self_tail')) for name in ['nano_vm','nanoisa','nvm2c']}
  args=list(args);args[0]=mapping.get(str(args[0]),args[0])
  result=super().run_command(*args,**kw)
  receipt.append({'argv':list(map(str,args)),'exit':result.returncode})
  return result
os.environ['CC']=cc
names=['test_self_tail_calls_publish_proved_array_authority','test_generic_direct_tail_calls_publish_proved_array_profile','test_depth_three_exact_leaves','test_nested_primitive_exact_leaves','test_five_scalar_elements']
suite=unittest.TestSuite(CurrentPublication(name) for name in names)
r=unittest.TextTestRunner(verbosity=2).run(suite)
(work/'paired-receipt.json').write_text(json.dumps(receipt,indent=2));raise SystemExit(not r.wasSuccessful())
