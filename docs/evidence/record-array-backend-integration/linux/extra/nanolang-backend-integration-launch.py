import os,sys,json,subprocess,hashlib,shutil,time
from pathlib import Path
base=Path(sys.argv[1]).resolve();darwin=sys.platform=='darwin';root=base/'source';report=base/'reports'
assert shutil.disk_usage(base).free>=2147483648
manifest=json.loads((base/'source.json').read_text());assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items())
env=dict(os.environ,LSAN_OPTIONS='',TMPDIR=str(base/'temporary'),RECORD_LLVM_NATIVE_OPTIMIZATIONS='O0,O2',PACKAGE_RUNNER=str(base/'nanolang-record-package-retain.py'),PYTHONDONTWRITEBYTECODE='1')
env.pop('CARRIER_PHASES',None)
if darwin:
 old=Path('/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921');clang=str(old/'tools/clang');assert '-isysroot /Applications/Xcode.app/' in Path(clang).read_text()
 llvm='/opt/homebrew/opt/llvm/bin/';cc='/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang'
 env.update(PATH=str(old/'tools')+':'+llvm+':/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',SDKROOT=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip(),LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib',NANO_FILE_RUNTIME_CFLAGS='-I/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk/usr/include/ffi',NMS_NATIVE_CLANG_FLAGS='',RECORD_LLVM_WASMTIME=str(old/'wasmtime43'),RECORD_LLVM_NODE='/opt/homebrew/bin/node')
 artifacts=json.loads((old/'nanolang-record-generated-75979-puck/ordinary-artifacts.json').read_text());baseline=next(str(Path(k).parent) for k in artifacts if k.endswith('/corpus/product-0000.c'))
 qualified=old/'nanolang-record-llvm-qualified-78042'
 extra={'real_clang':llvm+'clang','wasm_ld':'/opt/homebrew/opt/lld/bin/wasm-ld','ffi_header':'/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk/usr/include/ffi/ffi.h'}
else:
 llvm='/usr/local/bin/';clang=llvm+'clang';cc='/bin/gcc-13';baseline='/tmp/nano-record-array-generated-x9fej2n6/corpus';qualified=Path('/tmp/nanolang-record-llvm-qualified-78042')
 env.update(NMS_NATIVE_CLANG_FLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',RECORD_LLVM_WASMTIME='/home/jkh/.local/bin/wasmtime',RECORD_LLVM_NODE='/home/linuxbrew/.linuxbrew/bin/node')
 extra={'wasm_ld':llvm+'wasm-ld'}
env.update(CC=cc,CARRIER_SAN_CC=clang if darwin else cc,NMS_RUNTIME_CLANG=clang,NMS_RUNTIME_OPT=llvm+'opt',RECORD_LLVM_CLANG=clang,RECORD_LLVM_OPT=llvm+'opt',RECORD_LLVM_LLC=llvm+'llc',RECORD_LLVM_NM=llvm+'llvm-nm',RECORD_LLVM_C_BASELINE=baseline,CARRIER_EXTRA_TOOLS=json.dumps(extra),PACKAGE_BASELINE_HEADER=str(qualified/'obj/nanoisa/managed_runtime_ir.h'),PACKAGE_BASELINE_MANIFEST=str(qualified/'obj/nanoisa/managed_runtime_ir.json'))
record={'pin':manifest['pin'],'started':time.time(),'driver_sha256':hashlib.sha256((base/'nanolang-record-backend-integration-driver.py').read_bytes()).hexdigest(),'output_filesystem':str(base),'preflight_bytes':2147483648,'active_bytes':1610612736}
(base/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
with (base/'outer.log').open('wb') as log:
 p=subprocess.Popen([sys.executable,str(base/'nanolang-record-backend-integration-driver.py'),str(root),str(report)],env=env,stdout=log,stderr=subprocess.STDOUT);record['driver_pid']=p.pid;(base/'launch.json').write_text(json.dumps(record,indent=2)+'\n');code=p.wait()
record.update(returncode=code,finished=time.time());(base/'outer.json').write_text(json.dumps(record,indent=2)+'\n');sys.exit(code)
