from pathlib import Path
from unittest.mock import patch
import contextlib,copy,io,json,runpy
r=Path(__file__).resolve().parents[4];base=r/'docs/evidence/bootstrap-current-linux-09cab-20261122';manifest=base/'manifest.json';value=json.loads(manifest.read_text());seen=set();duplicate=None
for index,row in enumerate(value['files']):
 if row['sha256'] in seen:duplicate=index;break
 seen.add(row['sha256'])
assert duplicate is not None
original_read=Path.read_text;results=[]
for key,replacement in [('bytes',value['files'][duplicate]['bytes']+1),('compressed_sha256','0'*64),('artifact','objects/nonexistent.gz')]:
 bad=copy.deepcopy(value);bad['files'][duplicate][key]=replacement
 def modified(path,*args,**kwargs):
  if path==manifest:return json.dumps(bad)
  return original_read(path,*args,**kwargs)
 refused=False
 with patch.object(Path,'read_text',modified),contextlib.redirect_stdout(io.StringIO()):
  try:runpy.run_path(str(base/'read_archive.py'),run_name='__main__')
  except AssertionError:refused=True
 assert refused,key
 results.append(dict(control='duplicate-'+key,refused=True))
(base/'verification/descriptor-controls.json').write_text(json.dumps(dict(status='PASS',controls=results),indent=2)+'\n')
print('PASS three duplicate-descriptor corruption controls')
