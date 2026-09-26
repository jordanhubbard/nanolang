import subprocess,json,sys,time
from pathlib import Path
log=Path('/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/nanolang-record-llvm-78042-outer.log');status=log.with_suffix('.json')
record={'returncode':None,'error':None,'phase_limits':'driver:1800/14400/28800; child240','started':time.time()}
try:
 with log.open('wb') as output:
  result=subprocess.run([sys.executable,'/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/nanolang-record-llvm-78042-sdk-launch.py'],stdout=output,stderr=subprocess.STDOUT)
  record['returncode']=result.returncode
except Exception as error:record['error']=repr(error)
finally:
 record['finished']=time.time();status.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
sys.exit(record['returncode'] if record['returncode'] is not None else 125)
