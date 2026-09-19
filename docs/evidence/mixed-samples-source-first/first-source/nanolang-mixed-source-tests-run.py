import hashlib,json,os,pathlib,sys,time,unittest
from tests.test_source_borrow_emission import SourceBorrowEmission
from tests.test_source_mixed_samples import SourceMixedSamples
from tests.test_source_owned_string_fields import SourceOwnedStringFields
root=pathlib.Path.cwd(); out=pathlib.Path(sys.argv[1]);out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tools():return {str(p):sha(p) for p in [*SourceMixedSamples.emitters,*SourceMixedSamples.shadow_tools]}
report={'steps':[]};status=1
try:
 start=time.monotonic()
 SourceMixedSamples.setUpClass()
 initial=tools();(out/'producers-before.json').write_text(json.dumps(initial,indent=2)+'\n')
 report['setup_seconds']=round(time.monotonic()-start,3);report['work']=str(SourceMixedSamples.work)
 (out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
 # I share the same immutable freshly built producers across compiler configurations.
 def reuse(cls):
  cls.work=SourceMixedSamples.work;cls.emitters=SourceMixedSamples.emitters;cls.shadow_tools=SourceMixedSamples.shadow_tools
  cls.temporary=SourceMixedSamples.temporary
 def retain(cls):cls.temporary._finalizer.detach()
 for cls in (SourceMixedSamples,SourceBorrowEmission,SourceOwnedStringFields):
  cls.setUpClass=classmethod(reuse);cls.tearDownClass=classmethod(retain)
 focused=['test_original_all_shadows_routes_and_equal_metadata','test_alias_scope_empty_and_constructor_field_order','test_false_shadows_preserve_prior_publication','test_exact_type_nominal_and_profile_refusals']
 steps=[('focused-gcc','/usr/bin/gcc',unittest.TestSuite(SourceMixedSamples(n) for n in focused)),
        ('focused-clang','/tmp/nanolang-projection-clang',unittest.TestSuite(SourceMixedSamples(n) for n in focused)),
        ('old-source-profiles','/usr/bin/gcc',unittest.defaultTestLoader.loadTestsFromNames(['tests.test_source_borrow_emission','tests.test_source_owned_string_fields']))]
 for name,cc,suite in steps:
  os.environ['CC']=cc;start=time.monotonic();print('START',name,flush=True)
  with (out/(name+'.log')).open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2,failfast=True).run(suite)
  item={'name':name,'CC':cc,'status':0 if result.wasSuccessful() else 1,'testsRun':result.testsRun,'seconds':round(time.monotonic()-start,3)}
  report['steps'].append(item);(out/'status.json').write_text(json.dumps(report,indent=2)+'\n');print('END',item,flush=True)
  status=item['status']
  if status:break
finally:
 if hasattr(SourceMixedSamples,'temporary'):SourceMixedSamples.temporary._finalizer.detach()
 if hasattr(SourceMixedSamples,'emitters'):
  final=tools();(out/'producers-after.json').write_text(json.dumps(final,indent=2)+'\n')
  report['producer_identity_unchanged']=final==locals().get('initial',{})
  report['work']=str(SourceMixedSamples.work)
 report['status']=status;(out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(status)
