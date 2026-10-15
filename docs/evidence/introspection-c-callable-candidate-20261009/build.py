from pathlib import Path
import shlex,subprocess
root=Path('/private/tmp/nanolang-match-guards-20261009');out=Path('/private/tmp/nanolang-introspection-c-callables-20261009')
lines=(out/'build-commands.txt').read_text().replace('\\\n',' ').splitlines()
commands=[]
for line in lines:
 args=shlex.split(line)
 if '-c' in args and 'src/nanovirt/codegen.c' in args:
  args=[str(out/'codegen.c') if a=='src/nanovirt/codegen.c' else str(out/'codegen.o') if a=='obj/nanovirt/codegen.o' else a for a in args]
  args+=['-iquote',str(root/'src/nanovirt')];commands.append(args)
 elif '-c' in args and 'src/module.c' in args:
  args=[str(out/'module.c') if a=='src/module.c' else str(out/'module.o') if a=='obj/module.o' else a for a in args]; commands.append(args)
 elif '-o' in args and 'bin/nano_virt' in args:
  args=[str(out/'nano_virt') if a=='bin/nano_virt' else str(out/'codegen.o') if a=='obj/nanovirt/codegen.o' else str(out/'module.o') if a=='obj/module.o' else a for a in args];commands.append(args)
assert len(commands)==3,commands
for args in commands:
 print('COMMAND',args,flush=True)
 r=subprocess.run(args,cwd=root);print('EXIT',r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
