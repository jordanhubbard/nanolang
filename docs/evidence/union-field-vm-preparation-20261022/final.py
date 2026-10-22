from pathlib import Path
import os,subprocess,json,time,hashlib
root=Path.cwd();out=Path('/private/tmp/nanolang-union-vm-preparation-20261022');rows=[]
e=os.environ.copy();query=json.loads((root/'docs/evidence/union-field-image-20261022/target-env.json').read_text());runtime=json.loads((root/'docs/evidence/map-borrowed-invocation-20261022/final-environment.json').read_text())
e.update(query);e.update({k:v for k,v in runtime.items() if k.startswith('RECORD_ARRAY_VM_')});e['RECORD_ARRAY_VM_CFLAGS']='-Isrc -fsanitize=address,undefined -fno-sanitize-recover=all -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk/usr/include/ffi -I/opt/homebrew/Cellar/openssl@4/4.0.3/include';e['UNION_PREPARATION_MODULES']=query['UNION_IMAGE_MODULES'];e['LSAN_OPTIONS']=''
(out/'target-env.json').write_text(json.dumps({k:v for k,v in e.items() if k.startswith(('RECORD_ARRAY','UNION_'))},indent=2)+'\n')
def run(name,cmd):
 t=time.monotonic();p=subprocess.run(cmd,cwd=root,env=e,capture_output=True,timeout=240)
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr);rows.append(dict(name=name,command=cmd,status=p.returncode,seconds=time.monotonic()-t));(out/'final-status.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,p.returncode,round(rows[-1]['seconds'],3),p.stdout.decode()[-1800:],p.stderr.decode()[-1200:],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
run('permanent-preparation',['python3','-m','unittest','-f','-v','tests.test_union_field_preparation'])
run('image-recipes-proof-compatibility',['python3','-m','unittest','-f','-v','tests.test_union_field_image','tests.test_union_field_execution','tests.test_union_field_proof'])
for cc in ['/opt/homebrew/opt/llvm/bin/clang','/opt/homebrew/bin/gcc-16']:
 for mode,defs in [('default',[]),('union',['-DNANO_UNION_FIELD_PRIVATE_PREPARATION']),('combined',['-DNANO_UNION_FIELD_PRIVATE_PREPARATION','-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME','-DNANO_RECORD_MAP_PRIVATE_PREPARATION','-DNANO_RECORD_MAP_PRIVATE_RUNTIME','-DNANO_RECORD_MAP_PRIVATE_HEAP'])]:
  for dispatch,more in [('goto',[]),('switch',['-DNANO_NO_COMPUTED_GOTO'])]:
   run('strict-'+Path(cc).name+'-'+mode+'-'+dispatch,[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc',*defs,*more,'-fsyntax-only','src/nanovm/vm.c'])
 for observed in (False,True):
  run('strict-test-'+Path(cc).name+str(observed),[cc,'-std=c11','-D_DEFAULT_SOURCE','-Wall','-Wextra','-Werror','-Isrc','-DNANO_UNION_FIELD_PRIVATE_PREPARATION','-DNANO_NO_COMPUTED_GOTO',*(['-DVM_UF_WHITEBOX'] if observed else []),'-fsyntax-only','tests/nanoisa/test_union_field_preparation.c'])
for row in json.loads((root/'docs/evidence/map-string-consumer-20261022/budget-commands.json').read_text()):
 run(row['name'],[x.replace('/private/tmp/nanolang-map-string-consumer-20261022/budget-',str(out/'budget-')) for x in row['command']])
assert (out/'old-six-budget.stdout').read_bytes()==(root/'docs/evidence/map-string-consumer-20261022/old-six-budget.stdout').read_bytes()
print('six old budget rows byte-identical',flush=True)
selection=json.load(open(out/'retry/selection.json'))
for source in ['src/nanovm/vm.c','src/nanovm/vm_decode.c','src/nanovm/vm_dispatch.c','src/nanovm/heap.c','src/nanovm/heap_cycles.c','src/nanoisa/isa.c']:
 run('stack-'+Path(source).stem,selection['cc']+selection['flags']+selection['macros']+['-fstack-usage','-c',source,'-o',str(out/('stack-'+Path(source).stem+'.o'))])
run('make-n',['make','-n','-f','Makefile.gnu','test-union-field-preparation'])
