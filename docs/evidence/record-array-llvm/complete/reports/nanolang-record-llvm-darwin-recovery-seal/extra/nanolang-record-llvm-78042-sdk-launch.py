import hashlib,json,os,subprocess,sys,tarfile,shutil
from pathlib import Path
darwin=sys.platform=='darwin';host='puck' if darwin else 'linux'
root=Path('/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/nanolang-record-llvm-qualified-78042');archive=Path('/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/nanolang-record-llvm-78042-source.tar.gz');manifest=json.loads(Path('/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/nanolang-record-llvm-78042-source.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
assert shutil.disk_usage('/tmp').free>=3*1024**3,'I require3GiB before fresh setup'
root.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:tar.extractall(root,filter='data')
for name,digest in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
(root/'.qualification-tracked').write_text('\n'.join(manifest['files'])+'\n');(root/'.qualification-pin').write_text(manifest['pin']+'\n')
env=dict(os.environ,LSAN_OPTIONS='',CARRIER_PHASES='')
if darwin:
 env['TMPDIR']='/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/temporary'
 env['CARRIER_PHASES']=''
 llvm=Path('/opt/homebrew/opt/llvm/bin')
 env.update(CC='/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang',CARRIER_SAN_CC='/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/tools/clang',PATH='/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/tools:'+str(llvm)+':/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib',NMS_RUNTIME_CLANG='/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/tools/clang',NMS_RUNTIME_OPT=str(llvm/'opt'),RECORD_LLVM_WASMTIME='/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/wasmtime43',RECORD_LLVM_NODE='/opt/homebrew/bin/node',RECORD_LLVM_C_BASELINE=str(Path(next(p for p in json.loads(Path('/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/nanolang-record-generated-75979-puck/ordinary-artifacts.json').read_text()) if p.endswith('/corpus/product-0000.c'))).parent))
 env['SDKROOT']=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True,env=env).strip()
 ffi=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk/usr/include/ffi/ffi.h');assert ffi.is_file()
 env['NANO_FILE_RUNTIME_CFLAGS']='-I'+str(ffi.parent);env['CARRIER_EXTRA_TOOLS']=json.dumps({'libffi_header':str(ffi),'xcrun':'/usr/bin/xcrun','wasm_ld':'/opt/homebrew/opt/lld/bin/wasm-ld','selected_real_clang':str(llvm/'clang')})
else:
 env['CARRIER_PHASES']='setup,configuration,discovery,parity,ordinary-emission,ordinary-native,clang-ordinary-emission,clang-ordinary-native'
 llvm=Path('/usr/local/bin')
 env.update(CC='/bin/gcc-13',CARRIER_SAN_CC='/bin/gcc-13',NMS_RUNTIME_CLANG='/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/tools/clang',NMS_RUNTIME_OPT=str(llvm/'opt'),NMS_NATIVE_CLANG_FLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',RECORD_LLVM_WASMTIME='/home/jkh/.local/bin/wasmtime',RECORD_LLVM_NODE='/home/linuxbrew/.linuxbrew/bin/node',RECORD_LLVM_C_BASELINE='/tmp/nano-record-array-generated-x9fej2n6/corpus')
 env['CARRIER_EXTRA_TOOLS']=json.dumps({'wasm_ld':str(llvm/'wasm-ld')})
for name in ('clang','opt','llc','nm'):
 env['RECORD_LLVM_'+name.upper()]=('/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/tools/clang') if darwin and name=='clang' else str(llvm/('llvm-nm' if name=='nm' else name))
print(json.dumps({'pin':manifest['pin'],'source_files':len(manifest['files']),'root':str(root.resolve()),'free_bytes':shutil.disk_usage('/tmp').free,'selected':{k:v for k,v in env.items() if k.startswith('RECORD_LLVM_')}}),flush=True)
subprocess.run([sys.executable,'/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/nanolang-record-llvm-78042-driver.py',str(root),'/Users/jkh/nanolang-qualification/vm-effects-recovery-20260921/nanolang-record-llvm-78042-'+host],check=True,env=env)
