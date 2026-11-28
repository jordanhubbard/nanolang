"""I verify an unexecuted downstream gate and its live-predecessor refusal."""
from pathlib import Path
import ast,hashlib,json,subprocess
base=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert sha((base/'manifest.json').read_bytes())=='75f7423b0e349b6452d6a65a851a50d32d1c198890eed299eef97e4c5da1cfce'
m=json.loads((base/'manifest.json').read_bytes());assert not m['started'] and not m['accepted_component']
for name,h in m['files'].items():assert sha((base/name).read_bytes())==h
js=lambda name:json.loads((base/name).read_bytes())
p=js('PREPARED.json');r=js('refusal-control.json');red=js('live-foundation-refusal/result.json')
assert p['source_pin']==m['source_pin']=='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8' and not p['started'] and not p['release_acceptance']
assert p['method_count']==2 and p['full_source_routes']==5 and p['original_frame_cases_per_method']==1001
assert p['new_global_cases']==3688 and p['new_array_cases']==17056
assert p['limits']=='Original600 build, old8/600 frame batches, new32-or8KiB/60 batches, default10 shadow'
assert p['foundation_root']=='/var/tmp/nanolang-affine-typed-five-20261125-udz2o6c_' and p['foundation_receipt']=='affine-typed-five-linux-20261125'
assert r['terminal'] and r['accepted'] and r['exit']==1 and r['commands']==0 and not r['source_checkout_exists'] and r['matches_preparation']
assert red['terminal'] and not red['commands'] and not red['whole_release'] and not red['working_overlay']
assert sha((base/'setup.py').read_bytes())==r['source_sha256']==sha((base/'live-foundation-refusal/setup.py').read_bytes())
assert b'I require the terminal qualified five-method foundation before a new gate' in (base/'live-foundation-refusal/control.stderr').read_bytes()
setup=(base/'setup.py').read_text();admit=(base/'admit.py').read_text()
assert "foundation['observer_before']==foundation['observer_after']" in setup and "foundation['tests']==dict(tests=5,failures=0,errors=0,skips=0,passed=True)" in setup
assert "evidence=os.environ['NANOLANG_AFFINE_FOUNDATION_EVIDENCE_COMMIT']" in setup
assert "assert run('foundation-reader'" in setup and "'origin/feat/5.1-service-module-integration'" in setup
assert "assert sha(prior)==s['foundation_evidence']['result_sha256']" in admit
original=ast.parse((base/'original-cohort.py').read_text());save=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='save')
extra=[ast.dump(ast.parse(e).body[0]) for e in ["assert q['observer_before'][str(j/'cohort_journal.py')]==s['observer_source_sha256']","assert q['observer_before']==q['observer_after']"]]
class Normalize(ast.NodeTransformer):
 def visit_ImportFrom(self,node):return None if node.module=='cohort_journal' else node
 def visit_Assign(self,node):
  for t in node.targets:
   if isinstance(t,ast.Name) and t.id=='journal':return None
   if isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='q' and isinstance(t.slice,ast.Constant) and t.slice.value in ['observer_before','observer_after']:return None
  return self.generic_visit(node)
 def visit_Assert(self,node):return None if ast.dump(node) in extra else self.generic_visit(node)
 def visit_FunctionDef(self,node):return save if node.name=='save' else self.generic_visit(node)
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='journal' and node.func.attr=='row':node.func=ast.Name(id='dict',ctx=ast.Load())
  if isinstance(node.func,ast.Name) and node.func.id=='save':node.keywords=[k for k in node.keywords if k.arg!='metadata']
  return node
text=(base/'cohort.py').read_text();candidate=ast.parse(text)
calls=[n for n in ast.walk(candidate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save' and n.keywords]
assert len(calls)==3 and all(len(n.keywords)==1 and n.keywords[0].arg=='metadata' and isinstance(n.keywords[0].value,ast.Constant) and n.keywords[0].value.value is True for n in calls)
assert 'before_translation=True);save(metadata=True)' in text
for key,value in [('reference_argv','argv'),('array_reference_argv','array_argv')]:assert "q['"+key+"']="+value+";save(metadata=True);checked" in text
assert ast.dump(original)==ast.dump(Normalize().visit(candidate))
oldpath='docs/evidence/darwin-current-source-20261124/diagnostics/affine-array-values-c-linux-20261125/preparation/cohort.py'
assert (base/'original-cohort.py').read_bytes()==subprocess.check_output(['git','show','16687bbf02d947b3046eb064e6e159d060c93c42:'+oldpath],cwd=base)
qualified=json.loads(subprocess.check_output(['git','show','f588f39687b1863576472d5961baa71cef1a07f7:docs/evidence/darwin-current-source-20261124/diagnostics/affine-local-shadow-localization-linux-20261125/manifest.json'],cwd=base))
entry=next(r for r in qualified['files'] if r['path']=='observer-corrected/cohort_journal.py');assert sha((base/'cohort_journal.py').read_bytes())==entry['sha256']
for path in base.rglob('*.py'):compile(path.read_text(),str(path),'exec')
assert '257934' in (base/'foundation-terminal-templates/read_terminal.template.py').read_text()
print('PASS unexecuted source/producer/observer preparation, original two-method AST/limits, three metadata sites, published-success admission and live-foundation refusal before any command/checkout. Terminal templates remain unexecuted; no component/release acceptance.')
