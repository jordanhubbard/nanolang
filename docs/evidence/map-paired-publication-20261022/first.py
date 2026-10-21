import os,sys,json,shlex,hashlib
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root))
from tests.native_process import run_captured
out=Path(__file__).parent; agg=Path('/Users/jkh/Src/nanolang-51-harness')
config=dict(line.split('=',1) for line in (out/'closure.txt').read_text().splitlines())
cc=shlex.split(config['CC']);flags=shlex.split(config['CFLAGS']);ld=shlex.split(config['LDFLAGS'])
objects=[str(agg/x) for x in shlex.split(config['OBJECTS'])]
# The known Make concatenation repeats this archive; retain its final occurrence.
a=str(agg/'lib/libnano_compiler_inputs.a');objects=[x for i,x in enumerate(objects) if x!=a or a not in objects[i+1:]]
paths=objects+[str(agg/'bin'/x) for x in ('nano_virt','nano_vm','nanoisa','nvm2c')]
pins=lambda:{x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in paths}
(out/'pins-before.json').write_text(json.dumps(pins(),indent=2)+'\n');rows=[]
def run(name,args,build=False):
 p=run_captured([str(x) for x in args],cwd=root,env=os.environ,timeout=180,build=build)
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr)
 rows.append(dict(name=name,args=[str(x) for x in args],status=p.returncode));(out/'first-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert p.returncode==0,(name,p.stdout.decode(errors='replace'),p.stderr.decode(errors='replace'))
 return p.stdout
codegen=out/'codegen.o';facade=out/'nanoisa_facade.o'
run('codegen-build',cc+flags+['-c','src/nanovirt/codegen.c','-o',codegen],True)
run('facade-build',cc+flags+['-c','modules/nanoisa/nanoisa.c','-o',facade],True)
objects=[str(codegen) if x.endswith('/nanovirt/codegen.o') else str(facade) if x.endswith('/nanoisa/nanoisa_facade.o') else x for x in objects]
seed=out/'nano_virt';run('seed-link',cc+flags+objects+ld+['-o',seed],True)
source=root/'tests/nanoisa/fixtures/map_record_fields.nano';module=out/'ordinary.nvm'
run('ordinary-source',[seed,source,'--emit-nvm','--strip-debug','-o',module])
dump=run('ordinary-dump',[agg/'bin/nanoisa','dump',module]);(out/'ordinary.nasm').write_bytes(dump)
assert b'.ownership "03000000' in dump,'missing map authority'
run('ordinary-verify',[agg/'bin/nano_vm','--verify-only',module]);run('ordinary-vm',[agg/'bin/nano_vm',module])
run('ordinary-native',[agg/'bin/nvm2c',module,'-o',out/'ordinary.c'])
run('ordinary-native-build',cc+['-std=c11','-Wall','-Wextra','-Werror',out/'ordinary.c','-o',out/'ordinary'],True)
run('ordinary-native-execute',[out/'ordinary'])
after=pins();(out/'pins-after.json').write_text(json.dumps(after,indent=2)+'\n');assert json.loads((out/'pins-before.json').read_text())==after
print('first ordinary publication/verify/VM/native passed',flush=True)
