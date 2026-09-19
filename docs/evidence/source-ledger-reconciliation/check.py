"""I check only the bounded ledger/docs reconciliation, without executing artifacts."""
import hashlib,json,pathlib,re,subprocess
root=pathlib.Path(__file__).resolve().parents[3]
base=root/'docs/evidence/source-ledger-reconciliation'
tasks=json.loads((base/'task-states.json').read_text());states={r['id']:r['state'] for r in tasks}
for prefix in ('e64a267','c935734','68db1fb','c4351c7'):
 assert next(v for k,v in states.items() if prefix in k)=='completed'
for prefix in ('4be28','430220','a52262'):
 assert next(v for k,v in states.items() if prefix in k) not in ('completed','cancelled')
subprocess.run(['git','merge-base','--is-ancestor','f5abdb435bd0e9135ef25cb29459ec4ed9a9d9dd','HEAD'],cwd=root,check=True)
x=json.loads((base/'product-evidence-reference.json').read_text());data=subprocess.check_output(['git','show',x['commit']+':'+x['path']],cwd=root);assert hashlib.sha256(data).hexdigest()==x['sha256'];assert b'unchanged244' in data and b'exits2' in data
files=['docs/NANOISA_AFFINE_EXAMPLE_PREREQUISITES.md','docs/NANOISA_MIXED_SAMPLES_SOURCE.md','docs/evidence/source-ledger-reconciliation.md']
links=0
for name in files:
 f=root/name
 for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',f.read_text()):
  if '://' in target or target.startswith('#'):continue
  assert (f.parent/target.split('#',1)[0]).exists(),(name,target)
  links+=1
road=(root/'docs/ROADMAP.md').read_text()
for title in ('I restore the unchanged affine resource example','I lower exact ordinary Samples/flat FLOAT-array locals','I lower complete mixed scalar-leaf resource patterns','I select my intended Darwin sanitizer compiler'):
 assert '- [x] '+title in road
assert '- [ ] I exercise every embedded arithmetic provider' in road
assert '- [ ] I retain mutable FLOAT arrays inside affine Bundle owners' in road
subprocess.run(['git','diff','--check'],cwd=root,check=True)
print(f'I checked four completed boundaries, three open scopes, canonical ancestry, product evidence identity and {links} relative documentation links. No artifact was executed.')
