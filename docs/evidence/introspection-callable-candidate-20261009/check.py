from pathlib import Path
import subprocess,os,json
root=Path('/private/tmp/nanolang-match-guards-20261009'); out=Path('/private/tmp/nanolang-introspection-callables-20261009')
wrapper=out/'compiler';wrapper.write_text(f'#!/bin/sh\nexec {root}/bin/nano_vm {out}/compiler.nvm -- "$@"\n');wrapper.chmod(0o755)
lib=out/'probe.nano';lib.write_text('module callable_probe\npub struct Visible { value: int }\npub fn answer() -> int { return 7 }\nshadow answer { assert (== (answer) 7) }\n')
ops=[('is_unsafe','bool','false'),('has_ffi','bool','false'),('name','string','"callable_probe"'),('path','string',json.dumps(str(lib))),('function_count','int','1'),('struct_count','int','1'),('function_name','string','"answer"'),('struct_name','string','"Visible"')]
s=f'module {json.dumps(str(lib))} as probe\n'
for op,result,_ in ops:s+=f'extern fn ___module_{op}_callable_probe({"index: int" if op.endswith("_name") else ""}) -> {result}\n'
s+='let mut evaluations: int = 0\nfn index() -> int { set evaluations (+ evaluations 1) return 0 }\nshadow index { assert true }\n'
s+='fn returned() -> fn() -> int { return ___module_function_count_callable_probe }\nshadow returned { assert true }\n'
s+='fn invoke(f: fn() -> int) -> int { return (f) }\nshadow invoke { assert true }\n'
s+='fn main() -> int {\n'
for op,result,expected in ops:
 indexed=op.endswith('_name');s+=f'let value_{op}: fn({"int" if indexed else ""}) -> {result} = ___module_{op}_callable_probe\n'
 s+=f'assert (== (value_{op}{" (index)" if indexed else ""}) {expected})\n'
 if indexed:s+=f'assert (== (value_{op} -1) "")\nassert (== (value_{op} 1) "")\n'
s+='assert (== evaluations 2)\nlet count: fn() -> int = (returned)\nassert (== (invoke count) 1)\nreturn 0 }\nshadow main { assert (== (main) 0) }\n'
source=out/'all_values.nano';source.write_text(s)
commands=[[wrapper,source,'--emit-nvm','-o',out/'program.nvm'],[root/'bin/nano_vm','--verify-only',out/'program.nvm'],[root/'bin/nano_vm',out/'program.nvm'],[root/'bin/nvm2c',out/'program.nvm','-o',out/'program.c'],['/opt/homebrew/opt/llvm/bin/clang','-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',out/'program.c',root/'bin/nano_aot_runtime.o','-lm','-o',out/'program'],[out/'program']]
for cmd in commands:
 r=subprocess.run(list(map(str,cmd)),cwd=root,capture_output=True,text=True,timeout=120,env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1'))
 print('COMMAND',list(map(str,cmd)),'EXIT',r.returncode,r.stdout,r.stderr,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
