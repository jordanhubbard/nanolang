import os,sys,subprocess,json
from pathlib import Path
sys.path.insert(0,os.getcwd())
from tests.test_owned_global_source import OwnedGlobalSource
root=Path('/private/tmp/nanolang-selfhost-globals-initial');root.mkdir(exist_ok=True)
class Probe(OwnedGlobalSource):
 def source_route(self,source,accepted,diagnostics=None):
  w=root/self._testMethodName;w.mkdir(exist_ok=True);p=w/'main.nano';p.write_text(source)
  r=subprocess.run(['bin/nanoisa_emit',str(p),'-o',str(w/'main.nasm')],capture_output=True,text=True,timeout=30)
  (w/'emitter.log').write_text(r.stdout+r.stderr)
  if r.returncode==0:
   a=subprocess.run(['bin/nanoisa','asm',str(w/'main.nasm'),'-o',str(w/'main.nvm')],capture_output=True,text=True,timeout=30)
   (w/'assembler.log').write_text(a.stdout+a.stderr)
   if a.returncode==0:
    v=subprocess.run(['bin/nano_vm',str(w/'main.nvm')],capture_output=True,text=True,timeout=10)
    (w/'vm.log').write_text(v.stdout+v.stderr)
    print(self._testMethodName,accepted,'emit',r.returncode,'asm',a.returncode,'VM',v.returncode,flush=True)
   else:print(self._testMethodName,accepted,'asm',a.returncode,a.stderr[:200],flush=True)
  else:print(self._testMethodName,accepted,'emit',r.returncode,(r.stdout+r.stderr)[:180],flush=True)
for n in sorted(n for n in dir(Probe) if n.startswith('test_')):
 t=Probe(n);getattr(t,n)()
