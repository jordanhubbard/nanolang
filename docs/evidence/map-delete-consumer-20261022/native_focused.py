from pathlib import Path
import json,os,sys,subprocess,unittest
root=Path.cwd();sys.path.insert(0,str(root))
from tests.test_record_map_native import RecordMapNative
p=Path(__file__).parent;s=json.loads((p/'focused.json').read_text());old=Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-map-runtime-6gsrp07d');agg=Path('/Users/jkh/Src/nanolang-51-harness');sel=s['retained_selection']
objects=[str(old/(Path(x).stem+'.o')) for x in sel['providers'] if not x.endswith('/vm.c')]
objects +=[str(agg/x) if not Path(x).is_absolute() else x for x in sel['objects']]
objects=[x for x in objects if Path(x).stem!='nvm2c']
a=str(agg/'lib/libnano_compiler_inputs.a');objects=[x for i,x in enumerate(objects) if x!=a or a not in objects[i+1:]]
for stem in ['managed_array_shapes','verifier','ownership_contracts','heap']:objects=[str(p/('combined-'+stem+'.o')) if x==str(old/(stem+'.o')) else x for x in objects]
os.environ['RECORD_ARRAY_VM_LDFLAGS']=' '.join(sel['ldflags']);os.environ['RECORD_ARRAY_VM_OBJECTS']=' '.join(objects);os.environ['RECORD_ARRAY_VM_CC']=' '.join(sel['cc']);os.environ['RECORD_MAP_MODULES']='';os.environ['RECORD_ARRAY_VM_CFLAGS']='-fsanitize=address,undefined -fno-sanitize-recover=all -Isrc -Isrc/nanoisa'
class Focused(RecordMapNative):
 def driver(self,allocator=False):
  assert not allocator
  obj=p/'native-combined-vm.o'
  self.command('current-vm-build',self.cc+self.flags+['-DRA_ALLOC_WRAP','-include','tests/nanoisa/record_array_alloc.h','-c','src/nanovm/vm.c','-o',str(obj)],build=True)
  exe=self.work/'driver'
  self.command('driver-build',self.cc+self.flags+['-DMAP_NATIVE_WHITEBOX','tests/nanoisa/test_record_map_native.c','tests/nanoisa/record_array_alloc.c',str(obj),*objects,*self.ldflags,'-o',str(exe)],build=True)
  return exe
suite=unittest.TestSuite([Focused('test_native_semantics_and_allocation_prefixes')]);r=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not r.wasSuccessful())
