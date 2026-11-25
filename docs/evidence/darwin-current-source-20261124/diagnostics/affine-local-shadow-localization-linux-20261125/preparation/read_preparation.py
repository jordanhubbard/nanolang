"""I verify a changed-fixture gate's admission without claiming its terminal."""
from pathlib import Path
import ast,hashlib,json,subprocess
base=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
assert sha((base/'manifest.json').read_bytes())=='688dbd6d0da5227ab5bf2a1b80f0d439882a6584df2bb14c0e0e816fda87cc9d'
m=json.loads((base/'manifest.json').read_bytes());assert not m['accepted_component']
for name,h in m['files'].items():assert sha((base/name).read_bytes())==h
js=lambda name:json.loads((base/name).read_bytes())
s=js('result.json');a=js('admission-result.json');p=js('preflight-result.json');observation=js('launch-observation.json')
assert s['terminal'] and s['admitted_setup'] and s['all_guards_match'] and s['source_base']==m['source_pin']=='7b75651234c3235cd5a0ddd82d0a02c45fb3c8a8'
assert p['terminal'] and p['accepted'] and not p['source_checkout_started'] and len(p['commands'])==2 and all(c['exit']==0 and c['timeout']==180 for c in p['commands'])
assert sha((base/'cohort_journal.py').read_bytes())==p['observer_helper_sha256']
assert a['terminal'] and a['accepted'] and not a['other_jobs'] and a['free_bytes']>15*1024**3 and a['filesystem']!='nfs'
assert len(s['cpu_rounds'])==2 and all(r['share']>=.75 for r in s['cpu_rounds'])
assert observation['pid']==1170182 and observation['source_pin']==m['source_pin'] and not observation['terminal'] and not observation['accepted_component']
assert observation['producer_sha256']==s['producer_sha256']=='5a537570a4f6708212fd22c7ccedd8be69207e53a11db4e5bc579a443126b1c3'
assert observation['progress']['commands']>=1 and not observation['progress']['terminal']
for group in ['source','images','tools','hosts']:assert s[group+'_before']==s[group+'_after']
original=ast.parse((base/'original-cohort.py').read_text());save=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='save')
extra_asserts=[ast.dump(ast.parse(e).body[0]) for e in ["assert q['observer_before'][str(j/'cohort_journal.py')]==s['preflight_evidence']['observer_helper_sha256']","assert q['observer_before']==q['observer_after']"]]
class Normalize(ast.NodeTransformer):
 def visit_ImportFrom(self,node):return None if node.module=='cohort_journal' else node
 def visit_Assign(self,node):
  for t in node.targets:
   if isinstance(t,ast.Name) and t.id=='journal':return None
   if isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='q' and isinstance(t.slice,ast.Constant) and t.slice.value in ['observer_before','observer_after']:return None
  return self.generic_visit(node)
 def visit_Assert(self,node):return None if ast.dump(node) in extra_asserts else self.generic_visit(node)
 def visit_FunctionDef(self,node):return save if node.name=='save' else self.generic_visit(node)
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='journal' and node.func.attr=='row':node.func=ast.Name(id='dict',ctx=ast.Load())
  if isinstance(node.func,ast.Name) and node.func.id=='save':node.keywords=[k for k in node.keywords if k.arg!='metadata']
  return node
text=(base/'cohort.py').read_text();candidate=ast.parse(text)
metadata_calls=[n for n in ast.walk(candidate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save' and n.keywords]
assert len(metadata_calls)==4 and all(len(n.keywords)==1 and n.keywords[0].arg=='metadata' and isinstance(n.keywords[0].value,ast.Constant) and n.keywords[0].value.value is True for n in metadata_calls)
assert 'before_translation=True);save(metadata=True)' in text
for key,value in [('reference_argv','argv'),('scalar_reference_argv','scalar_argv'),('local_reference_argv','local_argv')]:assert "q['"+key+"']="+value+";save(metadata=True);checked" in text
assert ast.dump(original)==ast.dump(Normalize().visit(candidate))
for name in ['setup.py','admit.py','preflight.py','cohort_journal.py']:compile((base/name).read_text(),str(base/name),'exec')
assert js('collector-equivalence.json')['normalized_ast_equal']
receipt='docs/evidence/darwin-current-source-20261124/diagnostics/affine-foundation-shadow-failure-linux-20261125/manifest.json'
old=json.loads(subprocess.check_output(['git','show',p['evidence_commit']+':'+receipt],cwd=base));entry=next(r for r in old['files'] if r['path']=='comparison/cohort.py')
assert sha((base/'original-cohort.py').read_bytes())==entry['sha256']
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=base,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for name,h in s['source_after'].items():
 proc.stdin.write((m['source_pin']+':'+name+'\n').encode());proc.stdin.flush();parts=proc.stdout.readline().split();assert len(parts)==3 and parts[1]==b'blob';raw=proc.stdout.read(int(parts[2]));assert proc.stdout.read(1)==b'\n';assert sha(raw)==h,name
proc.stdin.close();assert proc.wait()==0
print('PASS source, published preflight, original producer, fresh admission, live launch observation and normalized original collector AST. No terminal, five-method acceptance or release acceptance.')
