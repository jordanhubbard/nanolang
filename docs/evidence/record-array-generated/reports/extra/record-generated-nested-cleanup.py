import subprocess,os,time,signal,json
from pathlib import Path
prefix='/tmp/nano-record-array-generated-ls4eyzd3/'
terminal=Path('/tmp/nanolang-record-generated-4fd32-linux/sanitizer-terminal.json')
out=Path('/tmp/nanolang-record-generated-4fd32-nested-cleanup.json')
seen={};events=[]
def live():
 rows={}
 for line in subprocess.check_output(['ps','-eo','pid=,pgid=,args='],text=True).splitlines():
  fields=line.strip().split(None,2)
  if len(fields)==3 and prefix in fields[2]:rows[int(fields[0])]={'pgid':int(fields[1]),'command':fields[2]}
 return rows
start=time.monotonic()
while not terminal.exists() and time.monotonic()-start<240:
 seen.update(live());time.sleep(1)
if terminal.exists():
 for sig in (signal.SIGTERM,signal.SIGKILL):
  rows=live()
  for pgid in sorted({r['pgid'] for r in rows.values()}):
   try:os.killpg(pgid,sig);events.append({'pgid':pgid,'signal':sig.name})
   except ProcessLookupError:pass
  deadline=time.monotonic()+5
  while time.monotonic()<deadline and live():time.sleep(.1)
remaining=live();out.write_text(json.dumps({'selected_path_prefix':prefix,'observed':seen,'outer_terminal_exists':terminal.exists(),'cleanup':events,'remaining':remaining,'all_selected_nested_processes_absent':not remaining},indent=2)+'\n')
print(out,not remaining)
