import os,sys,json,shlex,time
from pathlib import Path
r=Path.cwd();sys.path.insert(0,str(r));from tests.native_process import run_captured
from tests import test_record_array_vm as support
w=Path(__file__).parent;e=json.loads((r/'docs/evidence/map-copied-image-20261021/vm-environment.json').read_text());env=os.environ.copy();env['ASAN_OPTIONS']='detect_leaks=0'
cc=shlex.split(e['RECORD_ARRAY_VM_CC']);flags=[*shlex.split(e['RECORD_ARRAY_VM_CFLAGS']),'-std=c11','-D_DEFAULT_SOURCE','-O1','-UNDEBUG','-Isrc','-Isrc/nanoisa'];ld=shlex.split(e['RECORD_ARRAY_VM_LDFLAGS']);names={Path(p).stem for p in support.PROVIDERS};common=[p for p in shlex.split(e['RECORD_ARRAY_VM_OBJECTS']) if Path(p).stem not in names];receipts=[]
def run(name,args):
 t=time.monotonic();p=run_captured(args,cwd=r,env=env,timeout=240);(w/(name+'.stdout')).write_bytes(p.stdout);(w/(name+'.stderr')).write_bytes(p.stderr);receipts.append({'name':name,'argv':args,'exit':p.returncode,'seconds':time.monotonic()-t});(w/'legacy.json').write_text(json.dumps(receipts,indent=2));print(name,p.returncode,flush=True)
 if p.returncode:print(p.stderr.decode(),flush=True);sys.exit(p.returncode)
products=[]
for source in support.PROVIDERS:
 if source=='src/nanovm/heap.c':continue
 obj=w/('ordinary-'+Path(source).stem+'.o');run('ordinary-'+Path(source).stem,[*cc,*flags,'-c',source,'-o',str(obj)]);products.append(str(obj))
exe=w/'legacy-heap-allocation';run('legacy-heap-build',[*cc,*flags,'tests/nanovm/test_heap_allocation_failure.c',*products,*common,*ld,'-o',str(exe)]);run('legacy-heap-run',[str(exe)])
# The unchanged ordinary VM corpus includes the existing cycle collector and maps.
heap=w/'ordinary-heap.o';run('ordinary-heap',[*cc,*flags,'-c','src/nanovm/heap.c','-o',str(heap)])
exe=w/'ordinary-vm';run('ordinary-vm-build',[*cc,*flags,'tests/nanovm/test_vm.c',str(heap),*products,*common,*ld,'-o',str(exe)]);run('ordinary-vm-run',[str(exe)])
