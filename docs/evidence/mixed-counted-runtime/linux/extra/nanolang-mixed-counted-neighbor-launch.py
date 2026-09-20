import os,json,hashlib,subprocess,sys
from pathlib import Path
root=Path('/tmp/nanolang-mixed-counted-qualified-5f4f');old=Path('/tmp/nanolang-mixed-counted-5f4f-linux')
for rel,h in json.loads((old/'neighbor-managed_record_adapters-source-after.json').read_text()).items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
for path,entry in json.loads((old/'neighbor-managed_record_adapters-artifacts.json').read_text()).items():
 if Path(path).is_relative_to(root):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==entry['sha256'],path
selected=json.loads((old/'environment.json').read_text());env=dict(os.environ,**selected)
env.update(CC='/bin/gcc-13',CARRIER_SAN_CC='/bin/gcc-13',NMS_RUNTIME_CLANG='/usr/local/bin/clang',NMS_RUNTIME_OPT='/usr/local/bin/opt',NMS_NATIVE_CLANG_FLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',MC_NEIGHBOR_CC='/usr/local/bin/clang --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',LSAN_OPTIONS='',CARRIER_PHASES='configuration,neighbor-managed_record_adapters,neighbor-managed_array_copy_runtime,neighbor-managed_array_graphs,neighbor-managed_string_arrays,origin-query')
subprocess.run([sys.executable,'/tmp/nanolang-mixed-counted-neighbor-driver.py',str(root),'/tmp/nanolang-mixed-counted-5f4f-linux-neighbors'],env=env,check=True)
