#!/usr/bin/env python3
"""I inspect current production Clang evidence without executing native tools."""
from pathlib import Path
import gzip,hashlib,json,struct,subprocess
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
manifest=json.loads((HERE/'manifest.json').read_text())
files={};cache={}
for entry in manifest['files']:
    assert entry['path'] not in files
    artifact=entry['artifact']
    if artifact not in cache:
        packed=(HERE/artifact).read_bytes();cache[artifact]=(packed,gzip.decompress(packed))
    packed,raw=cache[artifact]
    assert (len(packed),hashlib.sha256(packed).hexdigest())==(entry['compressed_bytes'],entry['compressed_sha256'])
    assert (len(raw),hashlib.sha256(raw).hexdigest())==(entry['bytes'],entry['sha256'])
    files[entry['path']]=raw

def unique(suffix):
    found=[raw for name,raw in files.items() if name.startswith('download/') and name.endswith(suffix)]
    assert len(found)==1,(suffix,len(found))
    return found[0]

r=json.loads(unique('/capture-compiler/result.json'))
assert r['source_pin']=='9f3b69f5e3e2fe29ff82084b1cc7f9c9d2a876fb' and r['success'] and not r['errors']
run=json.loads(files['observer/current-clang-45036754742-run.json'])
assert run['id']==45036754742 and run['head_sha']==r['source_pin'] and run['conclusion']=='success'
assert r['all_source_hashes_unchanged']
assert r['native_identities_before']==r['native_identities_after']
assert r['native_closure_before']==r['native_closure_after']
assert r['selected_clang']['sha256']=='78e9dd2ccc9e88859b3a29109a44bfab00e3d6cb70f00f3daae063eed3d05251'
assert r['reports']['source']['discovered']==r['reports']['source']['ran']==128
assert r['reports']['source']['errors']==r['reports']['source']['failures']==0
assert r['reports']['source']['skipped']==json.loads(files['comparison/prior-complete-clang-tests.json'])['skipped']
assert len(r['reports']['source']['skipped'])==25
assert r['reports']['response']=={'discovered':1,'ran':1,'errors':0,'failures':0,'skipped':[]}
assert len(r['reports']['debug'])==9 and all(item['cases']==4 for item in r['reports']['debug'])
for item in r['reports']['debug']:
    report=json.loads(unique('/capture-compiler/'+item['label']+'.stdout'))
    assert len(report['cases'])==4 and all(not case['published_unit_aliases'] for case in report['cases'])
assert r['reports']['guide']['original_userguide_passed']
assert r['reports']['guide']['ordinary_userguide_environment']['inherited_shadow_override'] is None
assert r['reports']['guide']['ordinary_userguide_environment']['shadow_override'] is None
assert b'userguide-check: 40 snippet(s) passed' in unique('/capture-compiler/original-userguide.stdout')
assert r['cold_observation']['returncode']==0 and r['cold_observation']['compiler']==r['selected_clang']['path']
matrix=json.loads(unique('/capture-compiler/response-driver-grammar-matrix.json'))
assert len(matrix)==600 and all(row['captured_returncode']==0 and
    (row['native_returncode'],row['native_stdout'])==(row['replay_returncode'],row['replay_stdout']) for row in matrix)
assert len(r['objects'])==4
for index,obj in enumerate(r['objects']):
    assert obj['path'].endswith('.o') and len(obj['sha256'])==64
    comment=unique('/capture-compiler/object-producer-'+str(index)+'.stdout')
    assert b'clang version' in comment.lower() and b'GCC:' not in comment
build=unique('/.test_output/capture-assembler-checks/build.log')
assert len([line for line in build.splitlines() if line.startswith(b'clang ')])==6
for name,expected in [('test_module_generation_probe','c61fac6137641dfd5c0e3f2e6dd8ce0b098220e0b86b7e99274051f5533c5070'),('nano_as_capture.so','678c89bbd9a0b9cdfc01be19f281726139fef337303e5bb7639b4b1061a0ce0e')]:
    raw=unique('/capture-compiler/native/'+name)
    assert hashlib.sha256(raw).hexdigest()==expected and raw[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',raw,18)[0]==62
process=subprocess.Popen(['git','cat-file','--batch'],cwd=REPO,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,expected in r['source_inputs'].items():
    raw=files['executed-source/'+name];assert hashlib.sha256(raw).hexdigest()==expected
    process.stdin.write((r['source_pin']+':'+name+'\n').encode());process.stdin.flush()
    header=process.stdout.readline().decode().split();assert len(header)==3 and header[1]=='blob',name
    observed=process.stdout.read(int(header[2]));assert process.stdout.read(1)==b'\n' and observed==raw,name
process.stdin.close();assert process.wait(timeout=30)==0
print(json.dumps({'records':len(files),'Git_source_inputs':len(r['source_inputs']),'all_hashes_and_lengths_match':True,'native_Clang_objects_observed':4,'original_source_methods':128,'unchanged_skips':25,'response_cases':600,'response_stderr_matches':sum(row['native_stderr']==row['replay_stderr'] for row in matrix),'native_debug_cases':36,'ordinary_guide_snippets':40,'source_native_loader_identity_guards':True,'scope':'I read Python/Git data only; actual object-producer metadata is retained but raw intermediate .o files are not. This9f3 capture component is separate from later-source/full-platform/release acceptance.'},indent=2))
