"""I measure compiler scaling without changing my frozen bootstrap inputs."""
from pathlib import Path
import os,json,subprocess,time,hashlib
root=Path('/private/tmp/nanolang-match-guards-20261009')
work=Path(__file__).resolve().parent
results=[]
for count in (16,64,256):
 source=work/f'functions-{count}.nano'
 source.write_text('\n'.join(f'fn f{i}(x: int) -> int {{ return (+ x 1) }}\nshadow f{i} {{ assert (== (f{i} 2) 3) }}' for i in range(count))+'\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n')
 profile=work/f'profile-{count}.json';module=work/f'functions-{count}.nvm'
 command=[str(root/'bin/nano_vm'),'--profile-isa',str(profile),str(root/'bin/nanoc_seed.nvm'),'--',str(source),'--verbose','--emit-nvm','-o',str(module)]
 started=time.monotonic()
 with (work/f'compile-{count}.log').open('w') as log:
  result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=180,env={**os.environ,'CC':'false','NANO_CC':'false','NANOLANG_ROOT':str(root),'NANO_VM':str(root/'bin/nano_vm'),'NANOLANG_BOOTSTRAP_NO_CC':'1'})
 row={'functions':count,'source_bytes':source.stat().st_size,'elapsed_seconds':time.monotonic()-started,'exit_code':result.returncode,'command':command}
 if not result.returncode:
  verified=subprocess.run([str(root/'bin/nano_vm'),'--verify-only',str(module)],capture_output=True,text=True,timeout=30)
  row['verification_exit_code']=verified.returncode
 if profile.exists():row['profile']=json.loads(profile.read_text())
 results.append(row);(work/'results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps({k:v for k,v in row.items() if k not in ('command','profile')}),flush=True)
 if result.returncode:break
