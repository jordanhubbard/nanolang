import os,json,shlex,subprocess,hashlib,time,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tests.test_record_array_vm import PROVIDERS
root=Path.cwd();out=Path('/private/tmp/nanolang-union-vm-preparation-20261022/retry');out.mkdir(exist_ok=True)
env=json.load(open('docs/evidence/map-borrowed-invocation-20261022/final-environment.json'))
cc=['/opt/homebrew/opt/llvm/bin/clang'];flags=['-std=c11','-D_DEFAULT_SOURCE','-g','-O1','-Wall','-Wextra','-Werror','-Isrc','-Isrc/nanoisa','-fsanitize=address,undefined','-fno-sanitize-recover=all']
flags += [x for x in shlex.split(env['RECORD_ARRAY_VM_CFLAGS']) if x.startswith('-I') and x not in flags]
macros=['-DNANO_UNION_FIELD_PRIVATE_PREPARATION','-DNANO_NO_COMPUTED_GOTO']
names={Path(p).stem for p in PROVIDERS};objects=[p for p in shlex.split(env['RECORD_ARRAY_VM_OBJECTS']) if Path(p).stem not in names]
# Preserve only the final known compiler input archive occurrence.
objects=[p for i,p in enumerate(objects) if not (p.endswith('/libnano_compiler_inputs.a') and p in objects[i+1:])]
ld=shlex.split(env['RECORD_ARRAY_VM_LDFLAGS']);mods=shlex.split(json.load(open('docs/evidence/union-field-image-20261022/target-env.json'))['UNION_IMAGE_MODULES'])
(out/'selection.json').write_text(json.dumps(dict(cc=cc,flags=flags,macros=macros,providers=PROVIDERS,objects=objects,ldflags=ld,modules=mods),indent=2)+'\n')
paths=[*PROVIDERS,*objects,*mods,'src/nanovm/union_field_vm_prepare.inc','src/nanoisa/union_field_snapshot.inc','tests/nanoisa/test_union_field_preparation.c']
(out/'first-pins.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
status=[]
def run(name,args):
 (out/(name+'.argv.json')).write_text(json.dumps(args,indent=2)+'\n');start=time.monotonic()
 with open(out/(name+'.stdout'),'wb') as o,open(out/(name+'.stderr'),'wb') as e:r=subprocess.run(args,stdout=o,stderr=e,timeout=180,env={**os.environ,'LSAN_OPTIONS':''})
 status.append(dict(name=name,status=r.returncode,seconds=time.monotonic()-start));(out/'first-status.json').write_text(json.dumps(status,indent=2)+'\n')
 print(name,r.returncode,round(status[-1]['seconds'],3),flush=True)
 if r.returncode: print((out/(name+'.stderr')).read_text()[-5000:]);sys.exit(r.returncode)
for compiler,label in [(cc,'clang'),(['/opt/homebrew/bin/gcc-16'],'gcc')]:
 for selected in (False,True):
  for source in ['src/nanovm/vm.c','src/nanoisa/managed_array_shapes.c']:
   run('strict-'+label+('-private-' if selected else '-default-')+Path(source).stem,compiler+flags[:]+(macros if selected else ['-DNANO_NO_COMPUTED_GOTO'])+['-fsyntax-only',source])
for observed in (False,True):
 mode='observed' if observed else 'linked';products=[]
 for source in PROVIDERS:
  if observed and source=='src/nanovm/vm.c':continue
  obj=out/(mode+'-'+Path(source).stem+'.o')
  hook=['-DRA_ALLOC_WRAP','-include','tests/nanoisa/record_array_alloc.h'] if observed else []
  run(mode+'-'+Path(source).stem,cc+flags+macros+hook+['-c',source,'-o',str(obj)])
  products.append(str(obj))
 exe=out/(mode+'-preparation')
 run(mode+'-link',cc+flags+macros+(['-DVM_UF_WHITEBOX'] if observed else [])+['tests/nanoisa/test_union_field_preparation.c','tests/nanoisa/record_array_alloc.c',*products,*objects,*ld,'-o',str(exe)])
 run(mode+'-run',[str(exe),*(mods if not observed else [])])
 print((out/(mode+'-run.stdout')).read_text(),flush=True)
