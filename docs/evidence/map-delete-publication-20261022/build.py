from pathlib import Path
import os,sys,json,time,hashlib
root=Path.cwd();sys.path.insert(0,str(root));from tests.native_process import run_captured,native_compile_timeout
p=Path(__file__).parent;kind=sys.argv[1];assert kind in ['canonical','raw']
paths=[root/'bin/nanoc_c',root/'bin/nano_aot_runtime.o',Path('/private/tmp/nanolang-map-delete-consumer-20261022/nano_vm'),*root.glob('lib/*.a'),*root.glob('src_nano/**/*.nano'),root/'modules/nanoisa/module.json',root/'modules/nanoisa/nanoisa.c',root/'modules/nanoisa/nanoisa.h',root/'tests/nanoisa/fixtures/map_publication_driver.nano.txt']
paths +=[q for d in ['src/nanoisa','src/nanovm'] for q in (root/d).iterdir() if q.suffix in ['.c','.h','.inc']]
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();pins={str(q):sha(q) for q in paths};(p/(kind+'-input-pins.json')).write_text(json.dumps(pins,indent=2))
env=dict(os.environ,NANO_VM='/private/tmp/nanolang-map-delete-consumer-20261022/nano_vm')
for key,name in [('TMPDIR','compiler-tmp'),('NANO_BUILD_CACHE','module-cache')]:
 d=p/name;d.mkdir(exist_ok=True);env[key]=str(d)
source='src_nano/nanoc_v06.nano'
if kind=='raw':
 source=str(p/'raw-driver.nano');Path(source).write_bytes((root/'tests/nanoisa/fixtures/map_publication_driver.nano.txt').read_bytes())
args=[str(root/'bin/nanoc_c'),source,'-o',str(p/kind)];budget=native_compile_timeout(default=900);assert budget>=900
start=time.monotonic();result=run_captured(args,cwd=root,env=env,timeout=budget,build=False)
(p/(kind+'-build.stdout')).write_bytes(result.stdout);(p/(kind+'-build.stderr')).write_bytes(result.stderr)
(p/(kind+'-build.json')).write_text(json.dumps({'argv':args,'status':result.returncode,'seconds':time.monotonic()-start,'effective_timeout':budget,'environment':{k:env[k] for k in ['TMPDIR','NANO_BUILD_CACHE','NANO_VM']}},indent=2))
after={str(q):sha(q) for q in paths};(p/(kind+'-pins-after.json')).write_text(json.dumps(after,indent=2));assert pins==after
print(kind,'terminal',result.returncode,'seconds',time.monotonic()-start,flush=True);sys.exit(result.returncode)
