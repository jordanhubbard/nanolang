"""I verify preparation and the actual nonterminal six-method parent refusal."""
from pathlib import Path
import ast,hashlib,json,subprocess
b=Path(__file__).resolve().parent
raw=(b/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='549495400506f164d8531cdbc7f8096c458cef8fe6f497d61657c3164d6b65fc'
m=json.loads(raw);assert m['format']=='plain-controller-v1' and m['prepared_only'] and not m['accepted_component'] and not m['whole_release'] and len(m['files'])==9
for row in m['files']:
 raw=(b/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
w=json.loads((b/'preparation.json').read_bytes());q=json.loads((b/'nonterminal-parent-refusal.json').read_bytes())
assert w['prepared'] and not w['execution_started'] and not w['source_created'] and w['refusal_exit']==1 and w['commands']==0
assert q['terminal'] and q['commands']==[] and not q.get('admitted_setup',False)
assert w['methods']==1 and w['original_cases']==1001 and w['new_cases']==228 and w['expected_reports']==7374 and w['ordinary_seconds']==10
for name,h in w['controller_sha256'].items():
 raw=(b/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h;ast.parse(raw)
s=(b/'setup.py').read_text();assert s.index("assert six_parent['terminal']")<s.index('NANOLANG_CORRECTED_SIX_EVIDENCE_COMMIT')<s.index("run('complete-six-reader'")<s.index("root=j/'source';root.mkdir()")
assert "assert s['source_before']==six_parent['source_before']" in s
assert b'I require accepted terminal combined six-method evidence before source preparation' in (b/'nonterminal-parent-refusal.stderr').read_bytes()
c=(b/'cohort.py').read_text();assert 'new_methods=1' in c and 'new_cases=dict(preflight=228)' in c and 'suite.countTestCases()==1' in c and 'assert result.testsRun==1' in c
assert 'test_nanolang_translator_affine_preflight as preflight_ops' in c and 'NANOLANG_AFFINE_PREFLIGHT_REFERENCE' in c
for n in ['observer_before','observer_after','providers','object_inputs','all_guards_match','CohortJournal']:assert n in c
binding=json.loads((b/'reference-binding.json').read_bytes());assert '-Werror' in binding['reference_argv'] and binding['object_inputs']
source=subprocess.check_output(['git','show',w['source_base']+':'+binding['reference_source']],cwd=b);assert hashlib.sha256(source).hexdigest()==binding['reference_sha256']
print('PASS nine sealed preparation records, static syntax/Git reference binding and actual nonterminal six-method refusal; zero commands/source creation; no component acceptance')
