from pathlib import Path
import os,json,subprocess,time,hashlib
root=Path.cwd();out=Path('/private/tmp/nanolang-union-vm-preparation-20261022');env=os.environ.copy();env.update(json.load(open(out/'target-env.json')));env['LSAN_OPTIONS']='';rows=[]
def run(name,args):
 t=time.monotonic();p=subprocess.run(args,env=env,capture_output=True,timeout=240);elapsed=time.monotonic()-t
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr);rows.append(dict(name=name,command=args,status=p.returncode,seconds=elapsed));(out/'qualified-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(name,p.returncode,round(elapsed,3),p.stdout.decode()[-1000:],p.stderr.decode()[-1000:],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
run('qualified-permanent',['python3','-m','unittest','-f','-v','tests.test_union_field_preparation'])
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for obs in (False,True):
  run('qualified-test-'+Path(cc).name+str(obs),[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc','-DNANO_UNION_FIELD_PRIVATE_PREPARATION','-DNANO_NO_COMPUTED_GOTO',*(['-DVM_UF_WHITEBOX'] if obs else []),'-fsyntax-only','tests/nanoisa/test_union_field_preparation.c'])
select=json.load(open(out/'retry/selection.json'))
flags=[x for x in select['flags'] if not x.startswith(('-fsanitize','-fno-sanitize'))]
run('stack-vm-uninstrumented',select['cc']+flags+select['macros']+['-fstack-usage','-c','src/nanovm/vm.c','-o',str(out/'stack-vm-uninstrumented.o')])
run('default-vm-symbol-provider',select['cc']+select['flags']+['-DNANO_NO_COMPUTED_GOTO','-c','src/nanovm/vm.c','-o',str(out/'default-vm.o')])
run('default-vm-symbols',['nm','-g',str(out/'default-vm.o')]);assert 'vm_union_field_private_' not in (out/'default-vm-symbols.stdout').read_text()
run('private-vm-symbols',['nm','-g',str(out/'stack-vm.o')]);assert 'vm_union_field_private_prepare' in (out/'private-vm-symbols.stdout').read_text()
paths=['src/nanovm/vm.h','src/nanovm/heap.h','src/nanovm/vm.c','src/nanovm/union_field_vm_prepare.inc','src/nanovm/union_field_runtime_private.h','src/nanoisa/union_field_snapshot.inc','src/nanoisa/union_field_vm_image_private.h','tests/nanoisa/test_union_field_preparation.c','tests/test_union_field_preparation.py','Makefile.gnu']
(out/'qualified-source-pins.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
for p in ['src/nanovm/vm.h','src/nanovm/heap.h','src/nanoisa/managed_array_shapes.h']:
 old=subprocess.check_output(['git','show','75f811f684f0afddc11738bcbb4aa0b130bd961a:'+p]);assert old==Path(p).read_bytes()
print('old VM/heap/analysis public headers byte-identical',flush=True)
