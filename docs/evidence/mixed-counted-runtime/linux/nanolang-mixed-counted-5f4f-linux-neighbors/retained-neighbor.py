import os,sys,tempfile,unittest
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root));os.chdir(root)
work=Path(sys.argv[2]);work.mkdir(exist_ok=False)
print('I retain neighbor artifacts at '+str(work),flush=True)
class Retained:
 def __init__(self,suffix=None,prefix=None,dir=None,**kwargs):self.path=tempfile.mkdtemp(suffix=suffix or '',prefix=prefix or 'nano-retained-',dir=dir or work)
 def __enter__(self):return self.path
 def __exit__(self,*args):return False
tempfile.TemporaryDirectory=Retained
suite=unittest.defaultTestLoader.loadTestsFromName(sys.argv[3]);result=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite);raise SystemExit(not result.wasSuccessful())
