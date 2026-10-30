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
 rows.append(dict(name=name,args=[str(x) for x in args],status=p.returncode));(out/'controls-status.json').write_text(json.dumps(rows,indent=2)+'\n')
 assert p.returncode==0,(name,p.stdout.decode(errors='replace'),p.stderr.decode(errors='replace'))
 return p.stdout
objects=[str(out/'nanoisa_facade.o') if x.endswith('/nanoisa/nanoisa_facade.o') else x for x in objects if not x.endswith('/nanovirt/codegen.o') and not x.endswith('/nanovirt/main.o')]
run('controls-build',cc+flags+['tests/nanovirt/test_map_publication.c']+objects+ld+['-o',out/'controls'],True)
print(run('controls-run',[out/'controls',root/'tests/nanoisa/fixtures/map_record_fields.nano']).decode(),flush=True)
