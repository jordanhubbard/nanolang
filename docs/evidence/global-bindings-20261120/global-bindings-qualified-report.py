from pathlib import Path
import ast,hashlib,json
from tests.native_process import run_captured

root=Path.cwd()
directory=root/'.test_output/global-bindings-qualified'
directory.mkdir(exist_ok=True)
for node in ast.walk(ast.parse((root/'tests/test_eval_global_bindings.py').read_text())):
 if isinstance(node,ast.Constant) and isinstance(node.value,str):
  if node.value.startswith('module owned_globals\n'): (directory/'owner.nano').write_text(node.value)
  if node.value.startswith('module "owner.nano" as globals\n'): (directory/'main.nano').write_text(node.value)
prior=directory/'prior-native'
sentinel=b'I retain my prior product.\n'
prior.write_bytes(sentinel)
command=[root/'bin/nanoc_c',directory/'main.nano','-o',prior,'--llm-shadow-json',directory/'shadows.json']
run=run_captured(command,build=True,timeout=60,cwd=root)
report=json.loads((directory/'shadows.json').read_text())
data={'command':list(map(str,command)),'returncode':run.returncode,'stdout':run.stdout.decode(),'stderr':run.stderr.decode(),'prior_preserved':prior.read_bytes()==sentinel,'shadow_report':report,'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'main.nano',directory/'owner.nano')}}
(directory/'legacy-result.json').write_text(json.dumps(data,indent=2)+'\n')
assert report['completed'] and report['success'] and report['test_count']==7 and report['failures']==[]
assert run.returncode !=0 and data['prior_preserved']
print('I completed all seven qualified-global C-interpreter shadows; legacy C emission still refuses and preserves prior output.')
