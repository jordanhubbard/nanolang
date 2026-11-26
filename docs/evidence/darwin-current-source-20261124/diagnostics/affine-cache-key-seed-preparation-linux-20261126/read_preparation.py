"""I verify preparation and actual nonterminal-parent refusal only."""
from pathlib import Path
import ast,hashlib,json
b=Path(__file__).resolve().parent
raw=(b/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='6d00edba08ae2d8798eae4da1165ab5d3a9a638ca38206af5f02c0ad1e9deb91'
m=json.loads(raw);assert m['format']=='plain-controller-v1' and m['prepared_only'] and not m['accepted_component'] and not m['whole_release'] and len(m['files'])==9
for row in m['files']:
 raw=(b/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
w=json.loads((b/'preparation.json').read_bytes());q=json.loads((b/'result.json').read_bytes())
assert w['prepared'] and not w['execution_started'] and not w['source_created'] and w['refusal_exit']==1 and w['commands']==0
assert q['terminal'] and q['commands']==[] and not q.get('admitted_setup',False)
for name,h in w['controller_sha256'].items():
 raw=(b/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h;ast.parse(raw)
s=(b/'setup.py').read_text();assert s.index("assert triple['terminal']")<s.index("root=j/'source';root.mkdir()")
assert 'NANOLANG_REPORT1261_THREE_EVIDENCE_COMMIT' in s and 'beecf685b2ca9bc74791be2d8abf246689e01968' in s
assert b'I require accepted terminal corrected three-method evidence' in (b/'live-parent-refusal.stderr').read_bytes()
c=(b/'cohort.py').read_text();assert 'new_methods=1' in c and 'new_cases=dict(analysis_cache=3317)' in c and 'suite.countTestCases()==1' in c
print('PASS nine sealed preparation records, static syntax and actual parent refusal; zero commands/source creation; no component acceptance')
