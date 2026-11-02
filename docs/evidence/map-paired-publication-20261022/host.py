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
 rows.append(dict(name=name,args=[str(x) for x in args],status=p.returncode));(out/'host-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert p.returncode==0,(name,p.stdout.decode(errors='replace'),p.stderr.decode(errors='replace'))
 return p.stdout
config2=dict(line.split('=',1) for line in (out/'consumers.txt').read_text().splitlines())
items=[str(agg/x) for x in shlex.split(config2['nanoisa'])][1:]
items=[str(out/'nanoisa_facade.o') if x.endswith('/nanoisa/nanoisa_facade.o') else x for x in items]
raw=Path('/private/tmp/nanolang-map-publication-query-20261022/original-shadow.nvm')
run('current-shadow-verify',[out/'nano_vm','--verify-only',raw]);run('current-shadow-vm',[out/'nano_vm',raw])
run('host-build',cc+flags+['-I.', '-Itests/nanoisa',out/'host.c','tests/nanoisa/record_array_alloc.c']+items+ld+['-o',out/'host'],True)
print(run('host-run',[out/'host',out/'ordinary.nasm',raw]).decode(),flush=True)
