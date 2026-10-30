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
(out/'consumer-pins-before.json').write_text(json.dumps(pins(),indent=2)+'\n');rows=[]
def run(name,args,build=False):
 p=run_captured([str(x) for x in args],cwd=root,env=os.environ,timeout=180,build=build)
 (out/(name+'.stdout')).write_bytes(p.stdout);(out/(name+'.stderr')).write_bytes(p.stderr)
 rows.append(dict(name=name,args=[str(x) for x in args],status=p.returncode));(out/'consumer-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert p.returncode==0,(name,p.stdout.decode(errors='replace'),p.stderr.decode(errors='replace'))
 return p.stdout
config2=dict(line.split('=',1) for line in (out/'consumers.txt').read_text().splitlines())
all_inputs={}
for name in ('nanoisa','nano_vm','nvm2c'):
 items=[str(agg/x) for x in shlex.split(config2[name])]
 items=[x for i,x in enumerate(items) if x!=a or a not in items[i+1:]]
 all_inputs[name]={x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in items}
 run(name+'-link',cc+flags+items+ld+(shlex.split(config2['export']) if name=='nano_vm' else [])+['-o',out/name],True)
(out/'consumer-link-pins.json').write_text(json.dumps(all_inputs,indent=2)+'\n')
module=out/'ordinary.nvm'
dump=run('current-ordinary-dump',[out/'nanoisa','dump',module]);(out/'ordinary.nasm').write_bytes(dump)
assert b'.ownership "03000000' in dump
run('current-ordinary-verify',[out/'nano_vm','--verify-only',module]);run('current-ordinary-vm',[out/'nano_vm',module])
run('current-ordinary-native',[out/'nvm2c',module,'-o',out/'ordinary.c'])
run('current-ordinary-native-build',cc+['-std=c11','-Wall','-Wextra','-Werror',out/'ordinary.c','-o',out/'ordinary'],True)
run('current-ordinary-native-execute',[out/'ordinary'])
for name,inputs in all_inputs.items():
 assert inputs=={x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in inputs}
print('PASS unchanged ordinary with current isolated CLI providers',flush=True)
