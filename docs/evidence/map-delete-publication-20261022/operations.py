from pathlib import Path
import os,sys,json,hashlib,unittest,time
root=Path.cwd();sys.path.insert(0,str(root));from tests import test_map_field_operations as tests
p=Path(__file__).parent;fixture=p/'operation-root';fixture.mkdir(exist_ok=True);(fixture/'bin').mkdir(exist_ok=True)
if not (fixture/'tests').exists():(fixture/'tests').symlink_to(root/'tests',target_is_directory=True)
current=Path('/private/tmp/nanolang-map-delete-consumer-20261022');tools={n:current/n for n in ['nanoisa','nano_vm','nvm2c']};tools['nano_virt']=p/'nano_virt'
for name,target in tools.items():
 link=fixture/'bin'/name
 if not link.exists():link.symlink_to(target)
 assert link.resolve()==target.resolve()
os.environ['NANO_FLAT_ARRAY_BOUND_COMPILERS']=str(p/'nano_virt')+','+str(p/'canonical');os.environ['NANO_VM']=str(current/'nano_vm');tests.ROOT=fixture
paths=[*tools.values(),p/'canonical',root/'tests/test_map_field_operations.py',root/'tests/unit/test_map_field_operations.nano'];sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();pins={str(q):sha(q) for q in paths}
(p/'operations-config.json').write_text(json.dumps({'test_ROOT':str(fixture),'tests_link':str((fixture/'tests').resolve()),'tool_links':{str(fixture/'bin'/n):str(q) for n,q in tools.items()},'selectors':{k:os.environ[k] for k in ['NANO_FLAT_ARRAY_BOUND_COMPILERS','NANO_VM']},'pins':pins},indent=2))
suite=unittest.TestSuite([tests.MapFieldOperations(n) for n in ['test_paired_map_remove_preserves_void_and_aliases','test_paired_map_remove_refusals_preserve_output']]);start=time.monotonic();r=unittest.TextTestRunner(verbosity=2).run(suite);assert pins=={str(q):sha(q) for q in paths};(p/'operations-result.json').write_text(json.dumps({'tests':r.testsRun,'success':r.wasSuccessful(),'seconds':time.monotonic()-start},indent=2));sys.exit(not r.wasSuccessful())
