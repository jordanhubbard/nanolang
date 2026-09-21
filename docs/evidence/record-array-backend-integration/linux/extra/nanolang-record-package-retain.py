import hashlib,json,os,subprocess,tempfile,unittest
from pathlib import Path
artifacts=Path(os.environ['PACKAGE_ARTIFACTS']);artifacts.mkdir(exist_ok=False)
os.environ['TMPDIR']=str(artifacts);tempfile.tempdir=None
print('I retain package artifacts at '+str(artifacts),flush=True)
original_temp=tempfile.TemporaryDirectory
class RetainedDirectory(original_temp):
 def cleanup(self):self._finalizer.detach()
tempfile.TemporaryDirectory=RetainedDirectory
original_run=subprocess.run;serial=0
def recorded_run(*args,**kwargs):
 global serial
 serial+=1;prefix=artifacts/('command-%04d'%serial)
 argv=args[0] if args else kwargs.get('args');(prefix.with_suffix('.command.json')).write_text(json.dumps({'argv':[str(x) for x in argv],'timeout':kwargs.get('timeout')},indent=2)+'\n')
 try:
  result=original_run(*args,**kwargs)
  for name,value in [('stdout',result.stdout),('stderr',result.stderr)]:
   if value is not None:(prefix.with_suffix('.'+name)).write_bytes(value.encode() if isinstance(value,str) else value)
  prefix.with_suffix('.status.json').write_text(json.dumps({'returncode':result.returncode,'timeout':False})+'\n');return result
 except Exception as e:
  prefix.with_suffix('.status.json').write_text(json.dumps({'error':repr(e),'timeout':isinstance(e,subprocess.TimeoutExpired)})+'\n');raise
subprocess.run=recorded_run
from scripts import embed_managed_runtime as package
original_generate=package.generate;generations=[]
def retain_generate(*args,**kwargs):
 result=original_generate(*args,**kwargs);header,manifest,variants=result;number=len(generations)
 (artifacts/('package-%d.h'%number)).write_text(header);(artifacts/('package-%d.json'%number)).write_text(json.dumps(manifest,indent=2)+'\n')
 old_header=Path(os.environ['PACKAGE_BASELINE_HEADER']);assert old_header.read_text()==header,'I preserve the exact qualified header'
 old=json.loads(Path(os.environ['PACKAGE_BASELINE_MANIFEST']).read_text());new=json.loads(json.dumps(manifest))
 for value in (old,new):
  del value['sources']['scripts/embed_managed_runtime.py']
  for variant in value['variants'].values():
   del variant['boolean_abi']['ir'];del variant['boolean_abi']['ir_sha256']
 assert old==new,'Only generator identity and probe IR identity may differ'
 generations.append(hashlib.sha256(header.encode()).hexdigest());return result
package.generate=retain_generate
from tests import test_managed_runtime_package
suite=unittest.defaultTestLoader.loadTestsFromModule(test_managed_runtime_package);assert suite.countTestCases()==2
result=unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
(artifacts/'header-correspondence.json').write_text(json.dumps({'generations':generations,'qualified_header':os.environ['PACKAGE_BASELINE_HEADER'],'expected_generation_count':2},indent=2)+'\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
