import hashlib,json,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path.cwd()))
from tests.test_inline_owned_construction import InlineOwnedConstruction as Case
from tests.test_owned_record_patterns import OwnedRecordPatterns
out=pathlib.Path(sys.argv[1]);report={}
def snapshot(cls):
 paths=[*cls.emitters,*cls.shadow_tools]
 return {str(p):hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}
setup=Case.setUpClass.__func__;teardown=Case.tearDownClass.__func__
@classmethod
def before(cls):
 setup(cls);report['fixture_path']=str(cls.work);report['before']=snapshot(cls);out.write_text(json.dumps(report,indent=2)+'\n')
@classmethod
def after(cls):
 report['after']=snapshot(cls);report['unchanged']=report['after']==report['before'];out.write_text(json.dumps(report,indent=2)+'\n');teardown(cls)
Case.setUpClass=before;Case.tearDownClass=after
names=('test_ordered_children_and_all_selected_shadows','test_refusals_and_physical_local_budget','test_emitted_partial_construction_cleanup','test_float_local_unsafe_original_pattern','test_float_local_unsafe_control_flow','test_float_local_in_borrowed_control_flow','test_float_local_unsafe_refusals_preserve_publication','test_nested_owner_results_preserve_exact_transfer_and_order','test_nested_owner_result_refusals_preserve_output')
suite=unittest.TestSuite([*(Case(n) for n in names),OwnedRecordPatterns('test_unsafe_pattern_keeps_outer_shadow')])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() and report.get('unchanged') else 1)
