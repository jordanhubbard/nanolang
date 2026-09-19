import sys,re,json,hashlib,shutil
from pathlib import Path
for arg in sys.argv[1:]:
 report=Path(arg); dest=report/'retained-fixture-artifacts';dest.mkdir(exist_ok=True);entries={}
 for log in report.glob('*.log'):
  for directory in re.findall(r'I retain[^\n]* artifacts at (\S+)',log.read_text()):
   for p in Path(directory).rglob('*'):
    if p.is_file():
     h=hashlib.sha256(p.read_bytes()).hexdigest();target=dest/h
     if not target.exists():shutil.copyfile(p,target)
     entries[str(p)]={'sha256':h,'artifact':str(target)}
 (report/'retained-fixture-artifacts.json').write_text(json.dumps(entries,indent=2,sort_keys=True)+'\n')
 print(arg,len(entries))
