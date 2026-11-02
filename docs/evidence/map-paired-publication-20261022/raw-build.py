from pathlib import Path
import os,sys,json,time,hashlib,shutil
r=Path.cwd();sys.path.insert(0,str(r));from tests.native_process import run_captured,native_compile_timeout
w=Path(__file__).parent;root=Path('/Users/jkh/Src/nanolang-51-harness')
inputs=[root/'bin/nanoc_c',root/'bin/nano_aot_runtime.o',*root.glob('lib/*.a')]
pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
for p in inputs:
 target=r/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=pins[str(p)]:shutil.copy2(p,target)
env=os.environ.copy()
for name,part in [('TMPDIR','compiler-tmp'),('NANO_BUILD_CACHE','module-cache')]:
 p=w/part;p.mkdir(exist_ok=True);env[name]=str(p)
(w/'raw-host-source-pins.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [r/'modules/nanoisa/module.json',r/'modules/nanoisa/nanoisa.c',r/'src/nanoisa/managed_array_shapes.c',*r.glob('src/nanoisa/record_map*'),*r.glob('src/nanoisa/managed_record_map*')] if p.is_file()},indent=2))
(w/'raw-build-source-pins.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('src_nano/**/*.nano')},indent=2))
(w/'raw-build-inputs.json').write_text(json.dumps(pins,indent=2))
args=[str(r/'bin/nanoc_c'),str(w/'raw_driver.nano'),'-o',str(w/'raw_driver')];t=time.monotonic();p=run_captured(args,cwd=r,env=env,timeout=native_compile_timeout(900),build=True)
(w/'raw-build.stdout').write_bytes(p.stdout);(w/'raw-build.stderr').write_bytes(p.stderr);(w/'raw-build.json').write_text(json.dumps(dict(argv=args,exit=p.returncode,seconds=time.monotonic()-t,timeout=native_compile_timeout(900),environment={k:env[k] for k in ['TMPDIR','NANO_BUILD_CACHE']}),indent=2));print(p.stdout.decode(),p.stderr.decode(),flush=True);sys.exit(p.returncode)
