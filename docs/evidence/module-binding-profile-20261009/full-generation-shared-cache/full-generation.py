"""I compare isolated indexed lookup with the frozen full compiler generation."""
from pathlib import Path
import os,json,time,subprocess,signal,hashlib
work=Path(__file__).resolve().parent;root=Path('/private/tmp/nanolang-match-guards-20261009')
bootstrap=root/'obj/bootstrap-nanoisa/run-x8llgqn8'
guard=work/'guard-native-compiler'
guard.write_text((bootstrap/'guard-native-compiler').read_text().replace(str(bootstrap),str(work)));guard.chmod(0o755)
(work/'retained-host-inputs.json').write_bytes((bootstrap/'retained-host-inputs.json').read_bytes())
command=[str(work/'nano_vm'),'--profile-isa',str(work/'full-profile.json'),str(work/'compiler-indexed.nvm'),'--','src_nano/nanoc_v06.nano','--verbose','--emit-nvm','-o',str(work/'full-generation.nvm')]
report={'command':command,'source_commit':'af2b25da64679048f99c6d07b122eda132a54bc0','compiler_sha256':hashlib.sha256((work/'compiler-indexed.nvm').read_bytes()).hexdigest(),'timeout_seconds':1800}
start=time.monotonic()
with (work/'full-generation.log').open('w') as log:
 process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CC':str(guard),'NANO_CC':str(guard),'NANOLANG_ROOT':str(root),'NANO_MODULE_PATH':str(root/'modules'),'NANO_BUILD_CACHE':str(root/'obj/module_cache'),'NANO_VM':str(root/'bin/nano_vm'),'NANOLANG_BOOTSTRAP_NO_CC':'1'})
 report['pid']=process.pid;(work/'full-generation.json').write_text(json.dumps(report,indent=2)+'\n')
 try:report['exit_code']=process.wait(timeout=1800)
 except subprocess.TimeoutExpired:
  os.killpg(process.pid,signal.SIGKILL);process.wait();report['timed_out']=True
report['elapsed_seconds']=time.monotonic()-start;report['unexpected_native']=(work/'unexpected-native-compiler').exists()
if report.get('exit_code')==0:
 output=work/'full-generation.nvm';report['module_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
 verified=subprocess.run([str(root/'bin/nano_vm'),'--verify-only',str(output)],capture_output=True,text=True,timeout=30)
 report['verification_exit_code']=verified.returncode
(work/'full-generation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
