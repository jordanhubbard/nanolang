"""I bind every recorded stream-fault source input to my published source pin."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PIN='ddd185477'
sha=lambda data:hashlib.sha256(data).hexdigest()
def source(path):
    return subprocess.check_output(['git','show',PIN+':'+path],cwd=ROOT)

records=[]
for name in ('first','fallback'):
    report=json.loads((HERE/(name+'-report.json')).read_text())
    assert report['inputs_before']==report['inputs_after']
    for absolute,expected in report['inputs_before'].items():
        relative=Path(absolute).relative_to(ROOT).as_posix()
        assert sha(source(relative))==expected
        records.append({'report':name,'path':relative,'git_matches_reported_before_after':True})
eval_source=source('src/eval.c').decode()
helpers=eval_source[eval_source.index('static void shadow_json_escape('):eval_source.index('static bool shadow_write_json_file(')]
assert helpers.encode()==source('docs/evidence/shadow-json-span-comparison-20261109/original.inc')
assert helpers.encode()==source('docs/evidence/shadow-json-faults-20261109/fallback-candidate/original.inc')
print(json.dumps({'source_pin':subprocess.check_output(['git','rev-parse',PIN],cwd=ROOT,text=True).strip(),'recorded_source_entries':len(records),'unique_git_matched_sources':len({r['path'] for r in records}),'reference_helpers_match_pinned_runtime':True,'records':records,'native_tool_sdk_closure_qualified':False},indent=2))
