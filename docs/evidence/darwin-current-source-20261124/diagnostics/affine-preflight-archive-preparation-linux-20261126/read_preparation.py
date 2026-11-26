"""I verify a prepared archive collector, not a preflight qualification."""
from pathlib import Path
import hashlib,json,ast,subprocess
b=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
raw=(b/'manifest.json').read_bytes();assert sha(raw)=='c91c29a1e0fa38f0c3e46d851bb88827e0f325e7cb6fd0377239806b58227bb4'
m=json.loads(raw);assert m['format']=='plain-controller-v1' and m['prepared_only'] and not m['accepted_component'] and not m['whole_release'] and len(m['files'])==6
for row in m['files']:
 raw=(b/row['path']).read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
q=json.loads((b/'preparation-result.json').read_bytes());assert q['terminal'] and q['prepared_only'] and q['syntax'] and not q['accepted_component'] and not q['whole_release']
assert q['refusal_exit']==1 and not q['archive_created'] and q['compiler_commands']==0 and q['methods']==1 and q['expected_reports']==7374 and q['cases']==dict(original=1001,preflight=228)
for n,h in q['controller_sha256'].items():raw=(b/n).read_bytes();assert sha(raw)==h;ast.parse(raw)
s=(b/'package_terminal.py').read_text();assert s.index('assert terminal_file.is_file()')<s.index("out.mkdir()") and s.index("assert q['tests']==dict(tests=1")<s.index('out.mkdir()')
assert b'I require accepted terminal preflight evidence before archiving' in (b/'missing-terminal-refusal.stderr').read_bytes()
s=(b/'read_preflight.template.py').read_text()
for needle in ["m['methods']==1","q['tests']==dict(tests=1","q['new_cases']==dict(preflight=228)","==7 and len(q['providers'])==2","total==7374 and len(report_commands)+8",'committed and state==q',"q['source_before']==six['source_before']","sha(data('six-proof/manifest.json'))",'f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0','bf39b8692e5b95198377eef1c35683dc78404c94667024967a339cc8884b8a2c']:
 assert needle in s,needle
for name in ['source','images','tools','hosts','providers','observer','object_inputs','cohort-journal.jsonl']:assert name in s
f=json.loads((b/'fixture-binding.json').read_bytes());assert f['cases']==228 and f['compiler_commands']==0 and f['expected_stream_sha256']=='bf39b8692e5b95198377eef1c35683dc78404c94667024967a339cc8884b8a2c'
raw=subprocess.check_output(['git','show',f['source_commit']+':'+f['source_path']],cwd=b);assert sha(raw)==f['source_sha256']
print('PASS six sealed preparation records, exact fixture/Git binding, static syntax and actual missing-terminal refusal; zero compiler commands/archive creation; no preflight acceptance')
