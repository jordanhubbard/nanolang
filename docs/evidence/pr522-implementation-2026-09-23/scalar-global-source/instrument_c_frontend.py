from pathlib import Path
import shlex, subprocess, tempfile, os, sys, json
from tests.test_owned_scalar_global_source import cases
commands=Path(sys.argv[1]).read_text().replace('\\\n',' ').splitlines()
cc='/opt/homebrew/opt/llvm/bin/clang'
flags=['-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer']
with tempfile.TemporaryDirectory(prefix='pr522-global-c-instrument-') as d:
 work=Path(d);codegen=work/'codegen.o';checker=work/'typechecker.o';exe=work/'nano_virt'
 for line in commands:
  command=shlex.split(line)
  if not command or command[0]!='cc':continue
  if 'src/nanovirt/codegen.c' not in command and 'bin/nano_virt' not in command:continue
  command[0]=cc;command+=flags
  if '-c' in command:
   command[command.index('-o')+1]=str(codegen)
   command=[x for x in command if x not in ('-MMD','-MP')]
  else:
   command[command.index('-o')+1]=str(exe)
   command=[str(codegen) if x=='obj/nanovirt/codegen.o' else str(checker) if x=='obj/typechecker.o' else x for x in command]
  subprocess.run(command,check=True)
  if '-c' in command:
   subprocess.run(['src/typechecker.c' if x=='src/nanovirt/codegen.c' else str(checker) if x==str(codegen) else x for x in command],check=True)
 results=[]
 for name,source,accepted in cases():
  program=work/'source.nano';module=work/'source.nvm'
  program.write_text(source);module.write_bytes(b'prior module')
  result=subprocess.run([exe,program,'--emit-nvm','-o',module],capture_output=True,text=True,timeout=120,
    env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'})
  record=dict(case=name,accepted=accepted,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,prior_preserved=module.read_bytes()==b'prior module')
  print(json.dumps(record),flush=True)
  assert (result.returncode==0)==accepted,record
  assert accepted or record['prior_preserved'],record
  assert 'Sanitizer' not in result.stderr and 'runtime error:' not in result.stderr,record
 print('I passed 17 C scalar-global source cases with scoped ASan/UBSan/leak checks.')
