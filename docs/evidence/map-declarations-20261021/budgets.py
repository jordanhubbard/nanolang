import pathlib,subprocess,shlex,json,sys,time,hashlib
child=pathlib.Path.cwd();root=pathlib.Path('/Users/jkh/Src/nanolang-51-harness');work=pathlib.Path(__file__).parent
sys.path.insert(0,str(child));from tests.native_process import run_captured
from tests import test_record_array_vm as support
art=pathlib.Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-self-tail-ejpl9lh1')
text='include Makefile.gnu\n.PHONY: map-link\nmap-link:\n\t@echo $(sort $(NANOISA_OBJECTS) $(NANOVM_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS))\n\t@echo $(LDFLAGS)\n'
r=subprocess.run(['make','--no-print-directory','-s','-f','-','map-link'],input=text,text=True,cwd=root,capture_output=True,timeout=20);assert r.returncode==0,r.stderr
lines=r.stdout.splitlines();providers=list(dict.fromkeys([*support.PROVIDERS,'src/nanoisa/nvm_v2_convert.c','src/nanoisa/nvm_v2_module.c','src/nanoisa/nvm_format.c','src/nanoisa/nvm2c.c']))
names={pathlib.Path(p).stem for p in providers};common=[str(root/p) for p in shlex.split(lines[0]) if pathlib.Path(p).stem not in names]
common.sort(key=lambda p:p.endswith('.a'));ld=shlex.split(lines[1]);cc='/opt/homebrew/opt/llvm/bin/clang'
flags=['-std=c11','-D_DEFAULT_SOURCE','-g','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all','-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME','-Isrc','-Isrc/nanoisa']
receipt=[]
def run(name,args):
 t=time.monotonic();r=run_captured(list(map(str,args)),cwd=child,timeout=180);(work/(name+'.stdout')).write_bytes(r.stdout);(work/(name+'.stderr')).write_bytes(r.stderr);receipt.append({'name':name,'argv':list(map(str,args)),'exit':r.returncode,'seconds':time.monotonic()-t});(work/'budgets.json').write_text(json.dumps(receipt,indent=2));assert r.returncode==0,(name,r.stderr.decode());return r
products=[]
for p in providers:
 name=pathlib.Path(p).stem
 if name=='managed_array_shapes':continue
 if name in ('ownership_contracts','nvm_v2_layouts'):
  obj=work/(name+'-alloc.o');run(name+'-alloc-build',[cc,*flags,'-DRA_ALLOC_WRAP','-include','tests/nanoisa/record_array_alloc.h','-c',p,'-o',obj]);products.append(str(obj))
 else:products.append(str(art/(name+'.o')))
source=child/'docs/evidence/record-array-self-tail-20261021/old-budget.c'
run('old-six-budget-build',[cc,*flags,'-Itests/nanoisa','-DRA_WHITEBOX',source,'tests/nanoisa/record_array_alloc.c',*products,*common,*ld,'-o',work/'old-six-budget'])
r=run('old-six-budget',[work/'old-six-budget']);print(r.stdout.decode(),flush=True)
assert r.stdout.strip()==(child/'docs/evidence/record-array-self-tail-20261021/old-budget.stdout').read_bytes().strip()
(work/'budget-link-pins.json').write_text(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in [cc,*products,*common]},indent=2));print('EXACT SIX BUDGETS PASS',flush=True)
