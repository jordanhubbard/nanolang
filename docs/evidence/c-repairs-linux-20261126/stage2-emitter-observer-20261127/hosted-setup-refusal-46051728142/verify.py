"""I verify my setup refusal and corrected fixture controls as retained data."""
import ast
import hashlib
import json
from pathlib import Path
import re
import textwrap
import zipfile

root = Path(__file__).resolve().parent
raw = (root / 'manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == '3d600318dd3080ce3610b673d9508b2d1aeae6121e228ecdc370d8321dbbd9ed'
manifest = json.loads(raw)
assert not manifest['whole_release'] and not manifest['product_executed']
for name, record in manifest['files'].items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    data = (root / name).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
job = json.loads((root / 'first-hosted-job.json').read_text())
assert (job['id'], job['run_id'], job['head_sha'], job['conclusion']) == (
    140684444894, 46051728142, '31e4b6a2755c468ccc486bc37f262cab1b30012b', 'failure')
steps = {step['name']: step['conclusion'] for step in job['steps']}
assert steps['Select the original verified capture assembler'] == 'failure'
assert steps['Calibrate process observation and cache restoration'] == 'skipped'
assert steps['Observe the retained command at its original limit'] == 'skipped'
with zipfile.ZipFile(root / 'first-hosted-capture-artifact.zip') as archive:
    assert len(archive.namelist()) == 2 and archive.testzip() is None
    names = archive.namelist()
    identity = json.loads(archive.read(next(n for n in names if n.endswith('identity.json'))))
    assert identity['event_sha'] == job['head_sha']
    text = archive.read(next(n for n in names if n.endswith('controls.log'))).decode()
    assert re.search(r'Ran 8 tests in [0-9.]+s\n\nFAILED \(errors=2\)', text)
    assert text.count("FileNotFoundError: [Errno 2] No such file or directory:") == 2
    assert 'docs/evidence/capture-complete-hosted-20261122/manifest.json' in text
result = json.loads((root / 'fixture-check-result.json').read_text())
assert result['exit'] == 0 and result['fixture_count'] == 16
assert result['product_revision'] == 'd18fed86140b6a4cf983898ac32a079002f6b784'
assert not result['qualification'] and not result['target_stage2_executed']
assert re.search(r'Ran 8 tests in [0-9.]+s\n\nOK', (root / 'capture-controls.stderr').read_text())
assert len(json.loads((root / 'retained-fixtures.json').read_text())) == 16
workflow = (root / 'corrected-workflow.yml').read_text()
block = workflow.split('name: Materialize the pinned assembler profile inputs omitted by sparse checkout', 1)[1]
inline = textwrap.dedent(block.split("python3 - <<'PY'\n", 1)[1].split('\n          PY', 1)[0])
assert ast.dump(ast.parse(inline)) == ast.dump(ast.parse((root / 'fixture-materializer.py').read_text()))
print(json.dumps({'verified': True, 'setup_methods_failed': 2, 'setup_methods_run': 8,
                  'hosted_calibration_executed': False, 'target_stage2_executed': False,
                  'corrected_materialized_inputs': 16, 'corrected_setup_methods_passed': 8,
                  'product_execution': False, 'whole_release': False}))
