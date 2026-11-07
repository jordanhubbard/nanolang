from pathlib import Path
import subprocess,time,json,sys
p=Path(__file__).parent;rows=[]
for name in ['operations','old-bytes','strict']:
 t=time.monotonic();r=subprocess.run([sys.executable,str(p/(name+'.py'))],capture_output=True,timeout=180)
 (p/(name+'.log')).write_bytes(r.stdout+r.stderr);rows.append(dict(name=name,status=r.returncode,seconds=time.monotonic()-t));(p/'finish.json').write_text(json.dumps(rows,indent=2));print(name,r.returncode,flush=True);assert not r.returncode,(name,r.stdout,r.stderr)
