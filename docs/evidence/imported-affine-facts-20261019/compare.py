import ast,pathlib,subprocess,json,hashlib,shlex
root=pathlib.Path('/Users/jkh/Src/nanolang-51-checked-effects'); out=pathlib.Path('/private/tmp/nanolang-imported-affine-compare');out.mkdir(exist_ok=True)
old='/Users/jkh/Src/nanolang-51-harness/bin/nano_virt'; new='/private/tmp/nanolang-imported-affine-build/nano_virt'; dumper=str(root/'bin/nanoisa')
def run(args):return subprocess.run(list(map(str,args)),cwd=root,check=True,capture_output=True,text=True).stdout
source=root/'tests/nanoisa/fixtures/source_borrow_shared.nano'
for name,tool in [('old',old),('new',new)]: run([tool,source,'--emit-nvm','-o',out/(name+'.nvm')])
assert (out/'old.nvm').read_bytes()==(out/'new.nvm').read_bytes()
repro=pathlib.Path('/private/tmp/nanolang-imported-affine-repro')
run([new,repro/'main.nano','--emit-nvm','-o',out/'colliding-fixed.nvm'])
before=(repro/'actual.nasm').read_text();after=run([dumper,'dump',out/'colliding-fixed.nvm']);(out/'colliding-fixed.nasm').write_text(after)
def decode(text):
 lines=text.splitlines(); strings=[ast.literal_eval(l[8:]).encode('latin1') for l in lines if l.startswith('.string ')]
 facts=[(strings[int(l.split()[1])],strings[int(l.split()[2])]) for l in lines if l.startswith('.metadata ')]
 body=[];active=False
 for l in lines:
  if l.startswith('.function '):active=True;body.append(['.function']+shlex.split(l)[2:])
  elif active:body.append(l)
 return facts,body
assert decode(before)==decode(after)
result={'no_collision_sha256':hashlib.sha256((out/'new.nvm').read_bytes()).hexdigest(),'colliding_function_ordinals_signatures_instructions_and_metadata_payloads':'unchanged','new_wire_names':[shlex.split(l)[1] for l in after.splitlines() if l.startswith('.function ')]}
print(json.dumps(result,indent=2));(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
