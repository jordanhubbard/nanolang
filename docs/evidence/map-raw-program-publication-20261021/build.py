from pathlib import Path
import os,sys,json,time,hashlib
r=Path.cwd();sys.path.insert(0,str(r))
from tests.native_process import run_captured,native_compile_timeout
w=Path(__file__).parent
paths=[r/'bin/nanoc_c',r/'bin/nano_aot_runtime.o',r/'tests/nanoisa/fixtures/map_publication_driver.nano.txt',*r.glob('src_nano/**/*.nano'),r/'modules/nanoisa/nanoisa.c',r/'modules/nanoisa/module.json',*r.glob('src/nanoisa/record_map*'),r/'src/nanoisa/managed_array_shapes.c']
(w/'source-provider-pins.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},indent=2)+'\n')
env=dict(os.environ)
for key,name in [('TMPDIR','tmp'),('NANO_BUILD_CACHE','cache')]:
 p=w/name;p.mkdir(exist_ok=True);env[key]=str(p)
source=w/'driver.nano';source.write_bytes((r/'tests/nanoisa/fixtures/map_publication_driver.nano.txt').read_bytes())
args=[str(r/'bin/nanoc_c'),str(source),'-o',str(w/'driver')]
budget=native_compile_timeout(default=900);start=time.monotonic()
p=run_captured(args,cwd=r,env=env,timeout=budget)
(w/'build.stdout').write_bytes(p.stdout);(w/'build.stderr').write_bytes(p.stderr)
(w/'build.json').write_text(json.dumps(dict(argv=args,exit=p.returncode,seconds=time.monotonic()-start,timeout=budget,environment={k:env[k] for k in ['TMPDIR','NANO_BUILD_CACHE']}),indent=2)+'\n')
print('build terminal',p.returncode,flush=True);sys.exit(p.returncode)
