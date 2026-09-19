import json,hashlib,shutil,sys
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True);archive=out/'artifacts';archive.mkdir(exist_ok=True)
for name in sys.argv[2:]:
 src=Path(name);dest=out/src.name;dest.mkdir(exist_ok=True)
 for p in src.iterdir():
  if p.is_file() and p.suffix in ('.json','.log'):shutil.copyfile(p,dest/p.name)
 for inventory in src.glob('*-artifacts.json'):
  for item in json.loads(inventory.read_text()).values():
   sha=item['sha256'];p=Path(item['artifact']);target=archive/sha
   if not target.exists():
    assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
    shutil.copyfile(p,target)
