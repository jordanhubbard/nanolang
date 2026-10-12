from pathlib import Path
import tempfile,subprocess,unittest,json
from tests.test_resource_effect_frontend import ResourceEffectFrontend,HEADER,ROOT
rows=[]
def check(self,declaration,body,accepted):
 with tempfile.TemporaryDirectory() as directory:
  source=Path(directory)/'main.nano';out=Path(directory)/'main'
  source.write_text(HEADER+declaration+'\nfn main()->int{'+body+'}\nshadow main{assert true}\n')
  out.write_bytes(b'prior artifact')
  p=subprocess.run([ROOT/'bin/nanoc_c',source,'-o',out],capture_output=True,text=True,timeout=90)
  self.assertEqual(p.returncode,0 if accepted else 1,p.stdout+p.stderr)
  if accepted:
   r=subprocess.run([out],capture_output=True,text=True,timeout=10)
   self.assertEqual(r.returncode,7,r.stdout+r.stderr)
  else:self.assertEqual(out.read_bytes(),b'prior artifact')
  rows.append({'method':self.id(),'accepted':accepted})
ResourceEffectFrontend.check=check
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResourceEffectFrontend))
Path('/private/tmp/nl51-resource-effects/native-matrix-results.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(not r.wasSuccessful())
