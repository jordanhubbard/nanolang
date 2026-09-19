import pathlib,tempfile,shutil,unittest,json,hashlib
from tests.test_source_mixed_samples import SourceMixedSamples
w=pathlib.Path(tempfile.mkdtemp(prefix='nano-mixed-source-final-smoke-'));old=pathlib.Path('/tmp/nano-source-borrows-xaj6ccfr')
for f in old.iterdir():
 if f.name.endswith(('-emit','-shadows')):shutil.copy2(f,w/f.name)
def setup(cls):
 cls.work=w;cls.emitters=[pathlib.Path.cwd()/'bin/nanoisa_emit',w/'nanoc_stage1-emit',w/'nanoc_stage2-emit'];cls.shadow_tools=[w/(n+'-shadows') for n in ('nanoc_c','nanoc_stage1','nanoc_stage2')]
SourceMixedSamples.setUpClass=classmethod(setup);SourceMixedSamples.tearDownClass=classmethod(lambda cls:None)
r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([SourceMixedSamples('test_original_all_shadows_routes_and_equal_metadata')]))
print('retained',w,flush=True)
raise SystemExit(not r.wasSuccessful())
