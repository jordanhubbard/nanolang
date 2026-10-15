from pathlib import Path
import subprocess,json,time
w=Path(__file__).resolve().parent;r=Path('/private/tmp/nanolang-match-guards-20261009');rows=[]
for count in (16,64,256,1024):
 source=w/'by-function'/f'functions-{count}.nano';output=w/f'identical-{count}.nvm';start=time.monotonic()
 result=subprocess.run([str(r/'bin/nano_vm'),str(w/'compiler-indexed.nvm'),'--',str(source),'--emit-nvm','-o',str(output)],cwd=r,capture_output=True,text=True,timeout=180)
 (w/f'identical-{count}.log').write_text(result.stdout+result.stderr)
 row={'functions':count,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start,'byte_identical':result.returncode==0 and output.read_bytes()==(w/'by-function'/f'functions-{count}.nvm').read_bytes()}
 rows.append(row);print(row,flush=True);(w/'identical-results.json').write_text(json.dumps(rows,indent=2)+'\n')
