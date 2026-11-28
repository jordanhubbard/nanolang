from pathlib import Path
import hashlib,json,re,sys
base=Path(sys.argv[1]);out=Path(sys.argv[2]);candidate=base/'as246-candidate';producer=base/'fixed-as246'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
record=load(candidate/'result.json');build=load(producer/'build.json');source=load(candidate/'tests.json');response=load(candidate/'response-tests.json');matrix=load(candidate/'response-driver-grammar-matrix.json');cold=load(candidate/'cold-compiler-support/observation.json')
assert record['checkout_sources_restored'] and record['private_candidate_qualification_passed'] and record['fixed_identified_profile_qualification_passed'] and record['original_userguide_passed']
assert record['product_admission_changed'] is False and record['generic_gnu_246_admission_changed'] is False
assert source['ran']==source['discovered']==128 and response['ran']==response['discovered']==1
for report in (source,response):assert report['failures']==report['errors']==0
assert len(matrix)==600
for case in matrix:
 assert case['captured_returncode']==0
 assert case['native_returncode']==case['replay_returncode']
 assert case['native_stdout']==case['replay_stdout'] and case['native_stderr']==case['replay_stderr']
assert record['ordinary_userguide_environment']['shadow_override'] is None
assert record['ordinary_userguide_environment']['inherited_shadow_override'] is None
assert 'userguide-check: 40 snippet(s) passed' in (candidate/'original-userguide.stdout').read_text()
assert cold['returncode']==cold['supported_version_decision']==0
assert build['completed'] and build['source_sha256']=='d75a94f4d73e7a4086f7513e67e439e8fcdcbb726ffe63f4661744e6256b2cf2'
assert sha(producer/'input-scrub-before.c')=='40e7e14b5845e3f48b3be1cde0ea0b073f3e787864bdeb9d06dea9b545acfa07'
assert sha(producer/'input-scrub-fixed.c')==build['fixed_source_sha256']=='6ffe0ed82d02c8f92198b4c2b5200a8093b7eea0a07a96bf951eecabf021b036'
assert sha(producer/'bin/as')==build['assembler_sha256']==record['fixed_backend']['assembler_sha256']
assert sha(candidate/'candidate.patch')==record['candidate_patch_sha256']
rows=[]
for label in ('baseline','fixed'):
 text=(producer/(label+'-gas.sum')).read_text();rows.append([line for line in text.splitlines() if re.match(r'^(PASS|FAIL|XFAIL|XPASS|KFAIL|KPASS|UNSUPPORTED|UNRESOLVED|ERROR):',line)])
assert rows[0]==rows[1] and len(rows[1])==build['gas_cases']==2095
assert sum(line.startswith('PASS:') for line in rows[1])==build['gas_expected_passes']==2091
assert sum(line.startswith('UNSUPPORTED:') for line in rows[1])==4
assert not any(line.startswith(('FAIL:','XPASS:','UNRESOLVED:','ERROR:')) for line in rows[1])
assert len(build['witnesses'])==8 and all(w['original_source_identity']==w['expected'] for w in build['witnesses'])
for key,file in (('obj/test_module_generation_probe','test_module_generation_probe'),('bin/nano_as_capture.so','nano_as_capture.so')):assert sha(candidate/file)==cold['tool_sha256'][key]
as_key=next(k for k in cold['tool_sha256'] if k.endswith('/fixed-as246/bin/as'));assert cold['tool_sha256'][as_key]==build['assembler_sha256']
r=dict(source_pin=record['source_pin'],source_methods=source['ran'],source_conditional_skips=len(source['skipped']),response_methods=response['ran'],native_driver_equivalences=len(matrix),native_gas_passes=2091,native_gas_unsupported=4,all_2095_outcome_lines_equal=True,witnesses=8,ordinary_guide_snippets=40,shadow_override=None,cold_returncode=0,native_probe_sha256=sha(candidate/'test_module_generation_probe'),native_helper_sha256=sha(candidate/'nano_as_capture.so'),native_assembler_sha256=build['assembler_sha256'],candidate_patch_sha256=record['candidate_patch_sha256'],scope='I independently read the retained original reports, outcomes and actual native images. I do not infer current-source, production adoption or release acceptance.')
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
