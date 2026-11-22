from pathlib import Path
import hashlib,json,sys,re
base=Path(sys.argv[1]);out=Path(sys.argv[2]);candidate=base/'as246-candidate'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
record=load(candidate/'result.json');source=load(candidate/'tests.json');response=load(candidate/'response-tests.json');matrix=load(candidate/'response-driver-grammar-matrix.json');cold=load(candidate/'cold-compiler-support/observation.json')
assert record['checkout_sources_restored'] and record['clang_2118_complete_observation_passed']
assert record['product_admission_changed'] is False and record['gnu_assembler_admission_changed'] is False
assert source['ran']==source['discovered']==128 and response['ran']==response['discovered']==1
for report in (source,response):assert report['failures']==report['errors']==0
assert len(matrix)==600
for case in matrix:
 assert case['captured_returncode']==0
 assert case['native_returncode']==case['replay_returncode']
 assert case['native_stdout']==case['replay_stdout'] and case['native_stderr']==case['replay_stderr']
assert sha(candidate/'candidate.patch')==record['candidate_patch_sha256']
assert cold['returncode']==cold['supported_version_decision']==0 and cold['compiler']==record['selected_clang']['path']
assert cold['tool_sha256'][cold['compiler']]==record['selected_clang']['sha256']
comparisons=[]
for item in record['clang_native_equals_comparisons']:
 observed=load(candidate/(item['label']+'.stdout'));assert len(observed['cases'])==item['cases']==4
 for case in observed['cases']:
  assert case['published_unit_aliases']==[] and case['physical_object_identical'] and case['production_object_identical'] and case['generation_reused'] and case['answer']==42
  assert case['physical_debug_diff']==case['production_debug_diff']==case['warm_debug_diff']==''
 comparisons.extend(observed['cases'])
assert len(comparisons)==36
for key,file in (('obj/test_module_generation_probe','test_module_generation_probe'),('bin/nano_as_capture.so','nano_as_capture.so')):assert sha(candidate/file)==cold['tool_sha256'][key]
as_key=next(k for k in cold['tool_sha256'] if k.endswith('/fixed-as246/bin/as'))
assert cold['tool_sha256'][as_key]==record['fixed_backend']['assembler_sha256']==sha(base/'prior-fixed-clang/fixed-as246/bin/as')
assert record['fixed_clang_profile_qualification_passed'] and record['original_userguide_passed']
assert record['ordinary_userguide_environment']['shadow_override'] is None
assert record['ordinary_userguide_environment']['inherited_shadow_override'] is None
assert 'userguide-check: 40 snippet(s) passed' in (candidate/'original-userguide.stdout').read_text()
assert record['native_closure_before']==record['native_closure_after']
assert sha(candidate/'module_builder.c')=='971b242622730d2581bd5d576bec8dc7ce48a05c67e4917765d4f347f346bb09'
assert sha(candidate/'test_source_snapshots.py')=='7e80518ce26a983ec316fe161cbbf7c5a77a64724554ba3dc592f690ab7ee259'
assert sha(candidate/'candidate.patch')=='73d276fd14dae670b3634bfbabc128ea63ec5fe4bb2b897ca35855fc9a1b42d7'
assert response['skipped']==[]

r=dict(source_pin=record['source_pin'],source_methods=source['ran'],source_conditional_skips=len(source['skipped']),response_methods=response['ran'],native_driver_equivalences=len(matrix),native_equals_debug_cases=len(comparisons),cold_returncode=0,native_probe_sha256=sha(candidate/'test_module_generation_probe'),native_helper_sha256=sha(candidate/'nano_as_capture.so'),native_assembler_sha256=record['fixed_backend']['assembler_sha256'],candidate_patch_sha256=record['candidate_patch_sha256'],source_wall_seconds=float(re.search(r'Ran 128 tests in ([0-9.]+)s',(candidate/'original-tests.stderr').read_text()).group(1)),response_wall_seconds=float(re.search(r'Ran 1 test in ([0-9.]+)s',(candidate/'response-matrix.stderr').read_text()).group(1)),ordinary_guide_snippets=40,shadow_override=None,native_closure_stable=True,scope='I independently read the complete exact-source Clang/r1 consumer, all600 equivalences/36 debug cases/cold/ordinary40/default10 and actual retained images. Production/default CI and final platform/current-head acceptance remain separate.')
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
