from pathlib import Path
import sys,os,json,time,shutil,hashlib
r=Path.cwd();sys.path.insert(0,str(r));from tests.native_process import run_captured
w=Path(__file__).parent;target=r/'bin/nanoc_generic_record_qualified';shutil.copy2(w/'canonical',target)
env=os.environ.copy();env['NANOLANG_AFFINE_COMPILER_ROOT']=str(r/'bin');env['NANOLANG_AFFINE_COMPILERS']='nanoc_c,nanoc_generic_record_qualified';env['NANOLANG_ROOT']='/Users/jkh/Src/nanolang-51-harness'
args=[sys.executable,'-m','unittest','-f','-v','tests.test_affine_generic_identity'];t=time.monotonic();p=run_captured(args,cwd=r,env=env,timeout=900)
(w/'neighbors.stdout').write_bytes(p.stdout);(w/'neighbors.stderr').write_bytes(p.stderr);(w/'neighbors.json').write_text(json.dumps(dict(argv=args,exit=p.returncode,seconds=time.monotonic()-t,environment={k:v for k,v in env.items() if k.startswith('NANOLANG_')}),indent=2));print(p.stdout.decode(),p.stderr.decode(),flush=True);sys.exit(p.returncode)
