from pathlib import Path
import json,hashlib,sys
root=Path('/Users/jkh/Src/nanolang-51-owned-float-design');aggregate=Path('/Users/jkh/Src/nanolang-51-harness');sys.path.insert(0,str(aggregate))
from tests.native_process import run_captured
out=Path('/private/tmp/nanolang-primitive-current-budget');out.mkdir(exist_ok=True)
source=out/'baseline.c';source.write_text(Path('/private/tmp/nanolang-primitive-baseline.c').read_text().replace('/Users/jkh/Src/nanolang-51-nested-array',str(root)))
providers=['src/nanoisa/verifier.c','src/nanoisa/verifier_types.c','src/nanovm/vm_decode.c','src/nanoisa/ownership_contracts.c','src/nanoisa/nvm_v2_layouts.c','src/nanoisa/retained_layouts.c']
cc='/opt/homebrew/opt/llvm/bin/clang';flags=['-std=c11','-D_DEFAULT_SOURCE','-g','-O1','-Wall','-Wextra','-Werror','-Isrc','-Isrc/nanoisa']
old=json.loads(Path('/private/tmp/borrowed-empty-local-first-20261020/command.json').read_text())['argv'];excluded={Path(p).stem for p in providers}|{'managed_array_shapes'}
objects=[str(aggregate/p) for p in old if p.endswith(('.o','.a')) and Path(p).stem not in excluded]
ldflags=['-lm','-lcrypto','-lffi','-L/opt/homebrew/Cellar/openssl@4/4.0.3/lib']
commands=[]
def run(a):
 r=run_captured(a,cwd=root,text=True,timeout=180);commands.append({'argv':list(map(str,a)),'cwd':str(root),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert r.returncode==0,r.stderr;return r
files=[Path(cc),source]+[root/p for p in providers]+[Path(p) for p in objects]+[root/p for p in ('src/nanoisa/managed_array_shapes.c','src/nanoisa/record_array_origins.inc','src/nanoisa/ownership_array_fields.inc','tests/nanoisa/test_record_array_origins.c','tests/nanoisa/record_array_alloc.c','tests/nanoisa/record_array_alloc.h')]
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(out/'pins-before.json').write_text(json.dumps(pins,indent=2)+'\n')
products=[]
for provider in providers:
 obj=out/(Path(provider).stem+'.o');run([cc,*flags,'-DRA_ALLOC_WRAP','-include','tests/nanoisa/record_array_alloc.h','-c',provider,'-o',str(obj)]);products.append(str(obj))
exe=out/'baseline';run([cc,*flags,'-DRA_WHITEBOX',str(source),'tests/nanoisa/record_array_alloc.c',*products,*objects,*ldflags,'-o',str(exe)])
r=run([str(exe)]);(out/'baseline.log').write_text(r.stdout);after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};assert after==pins;(out/'pins-after.json').write_text(json.dumps(after,indent=2)+'\n');print(r.stdout)
