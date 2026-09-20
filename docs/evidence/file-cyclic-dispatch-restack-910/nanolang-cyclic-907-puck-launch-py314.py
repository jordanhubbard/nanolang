import json,os,pathlib,subprocess,sys,tarfile
root=pathlib.Path('/tmp/nanolang-cyclic-907-restack-3df-py314');root.mkdir()
with tarfile.open('/tmp/nanolang-cyclic-907-restack-3df.tar') as t:t.extractall(root,filter='data')
(root/'.qualification-tracked').write_bytes(pathlib.Path('/tmp/nanolang-cyclic-907-restack-tracked').read_bytes())
env=dict(os.environ);env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
def out(args):return subprocess.check_output(args,env=env,text=True).strip()
env['SDKROOT']=out(['/usr/bin/xcrun','--show-sdk-path']);env['CC']=out(['/usr/bin/xcrun','--find','clang']);env['CARRIER_SAN_CC']='/opt/homebrew/opt/llvm/bin/clang';env['LSAN_OPTIONS']='';env['LIBRARY_PATH']='/opt/homebrew/opt/openssl@3/lib';env['PKG_CONFIG_PATH']='/opt/homebrew/opt/libffi/lib/pkgconfig:/opt/homebrew/opt/openssl@3/lib/pkgconfig'
env['NMS_RUNTIME_CLANG']='/opt/homebrew/opt/llvm/bin/clang';env['NANO_FILE_RUNTIME_CFLAGS']=out(['/opt/homebrew/bin/pkg-config','--cflags','libffi'])
inc=out(['/opt/homebrew/bin/pkg-config','--variable=includedir','libffi']);env['CARRIER_EXTRA_TOOLS']=json.dumps({'xcrun':'/usr/bin/xcrun','pkgconfig':'/opt/homebrew/bin/pkg-config','libffi_header':inc+'/ffi.h','libffi_target_header':inc+'/ffitarget.h','owning_ld':out(['/usr/bin/xcrun','--find','ld'])})
r=subprocess.run([sys.executable,'/tmp/nanolang-cyclic-907-restack-driver.py',str(root),'/tmp/nanolang-cyclic-907-restack-puck'],env=env);sys.exit(r.returncode)
