from pathlib import Path
import hashlib,json,sys
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
as_key=next(k for k in cold['tool_sha256'] if k.endswith('/qualified-as242/bin/as'));assert cold['tool_sha256'][as_key]==record['qualified_backend']['sha256']==sha(base/'qualified-as242/bin/as')
r=dict(source_pin=record['source_pin'],source_methods=source['ran'],source_conditional_skips=len(source['skipped']),response_methods=response['ran'],native_driver_equivalences=len(matrix),native_equals_debug_cases=len(comparisons),cold_returncode=0,native_probe_sha256=sha(candidate/'test_module_generation_probe'),native_helper_sha256=sha(candidate/'nano_as_capture.so'),native_assembler_sha256=record['qualified_backend']['sha256'],candidate_patch_sha256=record['candidate_patch_sha256'],scope='I independently read complete original Clang reports, native equivalences/debug outcomes and actual native images. This cohort selects already-supported GNU2.42; fixedr1 external Clang and production promotion remain separate.')
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
