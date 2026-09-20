from pathlib import Path
import hashlib,json,subprocess,shutil,os
old=Path('/home/jkh/Src/nanolang-mixed-declaration-dd5a');new=Path('/home/jkh/Src/nanolang-mixed-declaration-5a367');reports=Path('/tmp/nanolang-mixed-declaration-dd5a-linux')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=json.loads((reports/'clang-ordinary-source-after.json').read_text())
for p,h in source.items():assert sha(old/p)==h,p
changed=[p for p,h in source.items() if sha(new/p)!=h];assert set(changed)=={'docs/ROADMAP.md','tests/nanoisa/test_ownership_declaration_projection.c'},changed
inputs=json.loads((reports/'clang-ordinary-inputs-before.json').read_text())
for p,v in inputs.items():assert sha(p)==v['sha256'],p
for d in ('obj','bin','lib'):
 if (old/d).is_dir():shutil.copytree(old/d,new/d)
for p,v in inputs.items():assert sha(new/Path(p).relative_to(old))==v['sha256'],p
proof={'original':'dd5a3ab637959c5cae1c3fd0a6212b516bb64265','corrected':'5a367e16e61587a424f7572cfd9bb68b1b4ee68f','sources_checked':len(source),'changed_paths':changed,'inputs_checked':len(inputs),'inputs':inputs}
Path('/tmp/nanolang-mixed-declaration-linux-reuse.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
env=dict(os.environ,CC='/usr/bin/gcc-13',CARRIER_SAN_CC='/usr/bin/gcc-13',LSAN_OPTIONS='',NMS_RUNTIME_CLANG='/usr/local/bin/clang',NMS_NATIVE_CLANG_FLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',CARRIER_PHASES='configuration,clang-ordinary,sanitizer,clang-sanitizer,ownership_contracts-build,ownership_contracts-run,ordinary_record_authority-build,ordinary_record_authority-run,nvm_v2_convert-build,nvm_v2_convert-run,nvm_v2_endtoend-build,nvm_v2_endtoend-run,old-array,affine-union,ownership-public')
for k in ('CFLAGS','LDFLAGS','NANOC','NANO_VM','NANO_NVM2C'):env.pop(k,None)
subprocess.run(['/usr/bin/python3','/tmp/nanolang-mixed-declaration-driver.py',str(new),'/tmp/nanolang-mixed-declaration-5a367-linux'],check=True,env=env)
