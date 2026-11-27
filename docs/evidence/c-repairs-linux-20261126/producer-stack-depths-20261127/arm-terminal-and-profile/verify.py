"""I verify retained ARM qualification and diagnostic bytes without executing products."""
from pathlib import Path
import argparse,gzip,hashlib,json,tempfile
p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);args=p.parse_args()
b=Path(__file__).resolve().parent;sha=lambda v:hashlib.sha256(v).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='a52dbed71e129d41766841cff0101ad66f0fe1421825abfa25acf0ad4fbef515'
m=json.loads(raw)
with tempfile.TemporaryDirectory(prefix='nano-arm-receipt-',dir=args.workspace) as d:
 root=Path(d)
 for row in m['files']:
  name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts
  output=root/name;output.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
  with output.open('wb') as stream:
   for key in row['chunks']:
    packed=(b/'objects'/(key+'.gz')).read_bytes();meta=m['chunks'][key]
    assert sha(packed)==meta['gzip_sha256'];data=gzip.decompress(packed)
    assert len(data)==meta['bytes'] and sha(data)==key
    stream.write(data);h.update(data);size+=len(data)
  assert size==row['size'] and h.hexdigest()==row['sha256']
 load=lambda n:json.loads((root/n).read_text())
 boot=load('qualification/bootstrap/result.json');native=load('qualification/native-suites/result.json')
 assert boot['exit']==0 and native['exit']==2 and not native['command_timeout']
 for result in [boot,native]:assert result['guards_match'] and result['before']==result['after']
 q=load('qualification/bootstrap-receipt.json')
 assert q['stage_timeout_seconds']==1800 and q['shadow_timeout_seconds']==30 and q['raw_stage1_stage2_equal']
 assert len(q['steps'])==17 and all(step['exit_code']==0 for step in q['steps'])
 for step in q['steps']:assert sha((root/'bootstrap-logs'/Path(step['log']).name).read_bytes())==step['log_sha256']
 assert (root/'products/nanoc_stage1.nvm').read_bytes()==(root/'products/nanoc_stage2.nvm').read_bytes()
 assert sha((root/'products/nanoc_stage2').read_bytes())=='8183b5cd681fbdd61913bdaec17cb5ced56124504ef53ce8edabdbf0a2b00864'
 text=(root/'qualification/native-suites/stderr').read_text()
 assert 'Ran 36 tests in 36.307s' in text and 'Ran 0 tests in 348.547s' in text and 'timed out after 300 seconds' in text
 d=load('profile/diagnostic/command/result.json')
 assert not d['qualification'] and d['exit_code']==0 and not d['command_timeout'] and d['deadline_seconds']==300
 assert load('profile/diagnostic/identities-before.json')==load('profile/diagnostic/identities-after.json')
 analysis=load('profile/sample-analysis.json');assert analysis['samples']==9059 and analysis['strlen_substr_split_lines_self_period']==69622920837
 assert analysis['total_period']==646422467268
 assert not m['whole_release'] and not m['diagnostic_qualification']
print('I verified original ARM17/36 and emitter setup timeout300, plus a separate9059-sample diagnostic; no complete ARM/release pass.')
