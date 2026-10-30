"""I compare completed sanitized compiler invocations with identical inputs."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[3]
out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=True)
seed=out/'nanoc_seed.nvm'
source=out/'parser_refactor.nano'
binaries={'A':root/'bin/nano_vm_decoded_baseline','B':root/'bin/nano_vm_decoded_view'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'seed_sha256':sha(seed),'source_sha256':sha(source),
         'binaries':{k:{'path':str(p),'sha256':sha(p)} for k,p in binaries.items()},
         'timeout_seconds':180,'runs':[]}
for number,variant in enumerate(('A','B','B','A','A','B')):
    output=out/'program.nvm'
    output.unlink(missing_ok=True)
    command=[str(binaries[variant]),str(seed),'--',str(source),'--emit-nvm','-o',str(output)]
    env=dict(os.environ,ASAN_OPTIONS='detect_leaks=0',UBSAN_OPTIONS='halt_on_error=1')
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    start=time.monotonic()
    process=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    timeout=False
    try:stdout,stderr=process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        timeout=True;os.killpg(process.pid,signal.SIGKILL);stdout,stderr=process.communicate()
    elapsed=time.monotonic()-start
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    prefix=f'{number}-{variant}'
    (out/(prefix+'.stdout')).write_bytes(stdout)
    (out/(prefix+'.stderr')).write_bytes(stderr)
    row={'variant':variant,'command':command,'returncode':process.returncode,'timeout':timeout,
         'wall_seconds':elapsed,'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
         'module_sha256':sha(output) if output.exists() else None}
    receipt['runs'].append(row)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if timeout or process.returncode:raise SystemExit(1)
    checked=subprocess.run([str(binaries[variant]),'--verify-only',str(output)],cwd=root,env=env,capture_output=True)
    (out/(prefix+'-verify.stdout')).write_bytes(checked.stdout)
    (out/(prefix+'-verify.stderr')).write_bytes(checked.stderr)
    assert checked.returncode==0
    executed=subprocess.run([str(binaries[variant]),str(output)],cwd=root,env=env,capture_output=True,timeout=30)
    (out/(prefix+"-execute.stdout")).write_bytes(executed.stdout)
    (out/(prefix+"-execute.stderr")).write_bytes(executed.stderr)
    assert executed.returncode==0
    row["verify_returncode"]=checked.returncode
    row["execute_returncode"]=executed.returncode
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
assert len({r['module_sha256'] for r in receipt['runs']})==1
