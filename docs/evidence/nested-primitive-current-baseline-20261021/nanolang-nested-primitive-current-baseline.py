from pathlib import Path
import ast,hashlib,json,sys
root=Path('/Users/jkh/Src/nanolang-51-harness');sys.path.insert(0,str(root))
from tests.native_process import run_captured
out=Path('/private/tmp/nanolang-nested-primitive-current-baseline');out.mkdir(exist_ok=True)
tree=ast.parse((root/'tests/test_ordinary_declaration_provenance.py').read_text())
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='test_nested_array_schema_retains_existing_unknown_boundary')
boolean=next(n.value for n in ast.walk(method) if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'struct Entries' in n.value)
sources={'bool-original':boolean,'float-original':(root/'tests/nanoisa/fixtures/nested_array_contexts.nano').read_text(),
'u8-record-probe':'''struct Entries { values: array<array<u8>> }
fn main() -> int { let entries: Entries = Entries { values: [[300], []] }
 assert (== (at (at entries.values 0) 0) 44)
 assert (== (array_length (at entries.values 1)) 0) return 0 }
shadow main { assert (== (main) 0) }
'''}
files=[root/'bin'/n for n in ('nano_virt','nanoc_stage1','nanoc_stage2','nano_vm','nanoisa','nvm2c')]+[root/'tests/test_ordinary_declaration_provenance.py',root/'tests/nanoisa/fixtures/nested_array_contexts.nano']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={str(p):sha(p) for p in files};(out/'pins-before.json').write_text(json.dumps(pins,indent=2)+'\n')
commands=[];results=[]
def run(a):
 r=run_captured(list(map(str,a)),cwd=root,text=True,timeout=180)
 commands.append({'argv':list(map(str,a)),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');return r
for name,text in sources.items():
 source=out/(name+'.nano');source.write_text(text)
 for compiler in ('nano_virt','nanoc_stage1','nanoc_stage2'):
  module=out/(name+'.'+compiler+'.nvm');row={'source':name,'compiler':compiler};results.append(row)
  r=run([root/'bin'/compiler,source,'--emit-nvm','-o',module]);row['compile']=r.returncode
  if r.returncode:continue
  row['verify']=run([root/'bin/nano_vm','--verify-only',module]).returncode
  row['vm']=run([root/'bin/nano_vm',module]).returncode
  r=run([root/'bin/nanoisa','dump',module]);(module.with_suffix('.nasm')).write_text(r.stdout)
  row['authority_present']=any(line.startswith('.ownership ') for line in r.stdout.splitlines())
  native=module.with_suffix('.c');r=run([root/'bin/nvm2c',module,'-o',native]);row['native_emit']=r.returncode
  if r.returncode:continue
  binary=module.with_suffix('.exe');r=run(['cc','-std=c11','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',native,'-lm','-o',binary]);row['native_build']=r.returncode
  if not r.returncode:row['native_run']=run([binary]).returncode
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');after={str(p):sha(p) for p in files};assert after==pins;(out/'pins-after.json').write_text(json.dumps(after,indent=2)+'\n');print(json.dumps(results,indent=2))
