import os,json,hashlib,pathlib,shutil,subprocess,sys
old=pathlib.Path(sys.argv[1]).resolve();new=pathlib.Path(sys.argv[2]).resolve();previous=pathlib.Path(sys.argv[3]).resolve();report=pathlib.Path(sys.argv[4]).resolve()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
source=json.loads((previous/'setup-source-before.json').read_text())
for p,h in source.items():assert sha(old/p)==h,p
if sys.platform=='darwin':
 assert not new.exists();shutil.copytree(old,new)
 subprocess.run(['/usr/bin/git','apply','/tmp/nanolang-file-indirect-flow-neighbors.patch'],cwd=new,check=True)
else:
 for name in ['obj','bin','lib']:
  if (old/name).exists():shutil.copytree(old/name,new/name)
products=json.loads((previous/'cyclic-neighbor-inputs-before.json').read_text());verified=0
for p,row in products.items():
 p=pathlib.Path(p)
 if p.is_relative_to(old):
  dest=new/p.relative_to(old);assert sha(dest)==row['sha256'],str(dest);verified+=1
changed={p for p,h in source.items() if sha(new/p)!=h}
assert changed=={'docs/ROADMAP.md','tests/nanoisa/test_file_cyclic.c'},changed
provenance={'original_source_verified':len(source),'copied_products_verified':verified,'changed_files':sorted(changed),'patch_sha256':sha('/tmp/nanolang-file-indirect-flow-neighbors.patch'),'production_and_new_fixture_unchanged':True}
pathlib.Path(str(report)+'-reuse.json').write_text(json.dumps(provenance,indent=2)+'\n')
env=dict(os.environ,CARRIER_PHASES='configuration,cyclic-neighbor,indirect-target-neighbor',LSAN_OPTIONS='')
if sys.platform=='darwin':
 env.update(PATH='/opt/homebrew/opt/llvm/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',CC='/usr/bin/clang',CARRIER_SAN_CC='/opt/homebrew/opt/llvm/bin/clang',SDKROOT=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip(),NANO_FILE_RUNTIME_CFLAGS='-I/opt/homebrew/opt/openssl@3/include',LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib')
else:env.update(CC='/usr/bin/gcc-13',CARRIER_SAN_CC='/usr/bin/gcc-13',NMS_NATIVE_CLANG_FLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13')
p=subprocess.run([sys.executable,'/tmp/nanolang-file-indirect-flow-driver.py',str(new),str(report)],env=env)
raise SystemExit(p.returncode)
