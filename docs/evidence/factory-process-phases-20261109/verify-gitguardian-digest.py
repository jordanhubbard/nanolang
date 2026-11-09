"""I verify the exact reported value against source bytes without echoing it."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

check_id = 128966401811
finding_commit = '994930b3804f950f3b9cfb568ea2c0e4b6a34435'
manifest_path = 'docs/evidence/async-source-contracts-20261109/source-869-apple/summary.json'
source_path = 'src_nano/compiler/token_value_bytes.nano'
raw = subprocess.check_output(['git', 'show', finding_commit+':'+manifest_path])
manifest = json.loads(raw)
keys = [name for name in manifest['source_creation']['inputs']['before'] if name.endswith('/'+source_path)]
assert len(keys) == 1
key = keys[0]
before = manifest['source_creation']['inputs']['before'][key]
after = manifest['source_creation']['inputs']['after'][key]
assert re.fullmatch('[0-9a-f]{64}', before)
source = subprocess.check_output(['git', 'show', manifest['source_pin']+':'+source_path])
assert hashlib.sha256(source).hexdigest() == before == after
line = json.loads('{'+raw.decode().splitlines()[77].strip().rstrip(',')+'}')
assert line == {key: before}
check = json.loads(subprocess.check_output(['gh', 'api',
    'repos/jordanhubbard/nanolang/check-runs/'+str(check_id)]))
assert check['name'] == 'GitGuardian Security Checks'
assert check['head_sha'] == 'd896dff6b33eea74f0739a9337562559b95dcee7'
reference = 'https://github.com/jordanhubbard/nanolang/commit/'+finding_commit+'#diff-'+hashlib.sha256(manifest_path.encode()).hexdigest()+'R78'
assert reference in check['output']['text']
report = {'observed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'check_run_id': check_id, 'check_name': check['name'],
    'check_conclusion': check['conclusion'], 'check_title': check['output']['title'],
    'check_head': check['head_sha'], 'finding_reference': reference, 'finding_commit': finding_commit,
    'finding_manifest': manifest_path, 'finding_line': 78,
    'source_commit': manifest['source_pin'], 'source_file': source_path,
    'reported_value_is_lowercase_hex_sha256': True,
    'actual_exact_source_blob_digest_matches_before_and_after': True,
    'verdict': 'I independently verify this occurrence as a public source-content checksum, not an authentication credential.',
    'dismissal_performed': False,
    'scope': 'I verify only this specific occurrence. I do not classify other GitGuardian findings or alter detector configuration, check conclusions, credentials or scan coverage.'}
Path(__file__).with_name('gitguardian-digest-verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
