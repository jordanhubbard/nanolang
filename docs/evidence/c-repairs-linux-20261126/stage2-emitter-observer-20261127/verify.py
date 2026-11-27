"""I verify selected terminal logs and observer controls without executing a product."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
root = args.evidence
sha = lambda data: hashlib.sha256(data).hexdigest()
raw = (root / 'manifest.json').read_bytes()
assert sha(raw) == 'e32cd7415edb0c99576c31756e4723500b1a007966cc4f940c76535422155395'
manifest = json.loads(raw)
assert manifest['version'] == 1 and manifest['hosted_observer_pending']
assert not manifest['whole_release'] and not manifest['whole_C_qualification']
for name, expected in manifest['files'].items():
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts
    data = (root / path).read_bytes()
    assert len(data) == expected['size'] and sha(data) == expected['sha256']

def log(name):
    text = (root / (name + '-job.log')).read_text()
    return '\n'.join(re.sub(r'\x1b\[[0-9;]*m', '', re.sub(
        r'^\d{4}-\d\d-\d\dT\S+Z ?', '', line)).strip() for line in text.splitlines())


jobs = [('older-linux-bootstrap', 140652282080, 46042142609,
         'd18fed86140b6a4cf983898ac32a079002f6b784'),
        ('linux-bootstrap', 140656541879, 46043406093,
         'cd6bf9fbfff108f3dffb541ae4c266e8d6b83334'),
        ('darwin-bootstrap', 140656542299, 46043406093,
         'cd6bf9fbfff108f3dffb541ae4c266e8d6b83334')]
commands = []
for name, job, run, revision in jobs:
    data = json.loads((root / (name + '-job.json')).read_text())
    assert (data['id'], data['run_id'], data['head_sha']) == (job, run, revision)
    assert data['status'] == 'completed' and data['conclusion'] == 'failure'
    steps = {step['name']: step['conclusion'] for step in data['steps']}
    assert steps['Execute the original bootstrap with its default limits'] == 'success'
    assert steps['Check managed STRING publication and exact alias retention'] == 'success'
    assert steps['Execute the original native compiler and emitter suites'] == 'failure'
    assert steps['Verify source and selected tool guards'] == 'success'
    text = log(name)
    assert re.search(r'^Ran 36 tests in [0-9.]+s\n\nOK$', text, re.MULTILINE)
    assert re.search(r'^Ran 0 tests in [0-9.]+s\n\nFAILED \(errors=1\)$', text, re.MULTILINE)
    if name.startswith('darwin'):
        assert 'AssertionError: nanoc_c exited 1:' in text
        assert 'I stopped shadow execution after 10 seconds.' in text
    else:
        matches = re.findall(r"^subprocess.TimeoutExpired: Command '(\[.*\])' timed out after 300 seconds$",
                             text, re.MULTILINE)
        assert len(matches) == 1
        command = ast.literal_eval(matches[0])
        assert command[:3] == ['/home/runner/work/nanolang/nanolang/bin/nanoc_stage2',
                               '/home/runner/work/nanolang/nanolang/tests/selfhost_shadow_emitter.nano', '-o']
        commands.append(command)
cache = json.loads((root / 'linux-cache-job.json').read_text())
assert (cache['id'], cache['run_id'], cache['status'], cache['conclusion']) == (
    140656542401, 46043406093, 'completed', 'success')
controls = json.loads((root / 'controls-result.json').read_text())
assert controls['exit'] == 0 and controls['tests_run'] == 16
assert not controls['qualification'] and not controls['product_execution']
assert controls['native_process_controls_executed']
assert re.search(r'\nRan 16 tests in [0-9.]+s\n\nOK\n',
                 (root / 'final-workflow-controls.stderr').read_text())
for name, expected in controls['copied_controller_sources_match'].items():
    assert sha((root / 'source' / name).read_bytes()) == expected
original = json.loads((root / 'original-command-check.json').read_text())
assert original['actual_retained_log_accepted'] and not original['qualification']
assert original['argv'] == commands[0]
assert sha((root / 'restore-source/scripts/ci_darwin_shadow_lifecycle.py').read_bytes()) == (
    '5693f86ea998efff0abde49284d362c79c21556cabc6739647438878ce81771f')
print(json.dumps({'verified': True, 'selected_files': len(manifest['files']),
                  'original_bootstrap_and_36_native_methods_passed': 3,
                  'linux_stage2_300_second_setup_refusals': 2,
                  'darwin_cseed_10_second_setup_refusals': 1,
                  'emitter_methods_completed_in_these_jobs': 0,
                  'linux_cache_job_passed': True, 'observer_tests_passed': 16,
                  'hosted_observer_pending': True, 'product_execution': False,
                  'whole_C_qualification': False, 'whole_release': False}))
