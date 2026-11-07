import json,subprocess,hashlib,platform,time
from pathlib import Path
cc='/tmp/toolchain/usr/bin/x86_64-linux-gnu-gcc-14'
print(subprocess.check_output([cc,'--version'],text=True),flush=True)
print(platform.platform(),flush=True)
base=[cc,'--sysroot=/tmp/toolchain','-Wall','-Wextra','-Werror','-std=c99','-O3','-g','-Isrc','-Isrc/nanoisa','-D_GNU_SOURCE','-D_FORTIFY_SOURCE=2']
results=[]
for label,source in [('current','tests/nanoisa/test_disasm_roundtrip.c'),('old','/tmp/replay/tests/nanoisa/test_disasm_roundtrip.c')]:
 for san in (False,True):
  args=[*base,*(['-fsanitize=address,undefined','-fno-omit-frame-pointer'] if san else []),'-S',source,'-o','/dev/null']
  t=time.monotonic();r=subprocess.run(args,capture_output=True,text=True,timeout=90)
  results.append({'label':label,'sanitizers':san,'argv':args,'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'returncode':r.returncode,'stderr':r.stderr,'seconds':time.monotonic()-t})
  print(json.dumps(results[-1]),flush=True)
Path('/tmp/result.json').write_text(json.dumps(results,indent=2))
assert all(r['returncode']==0 for r in results if r['label']=='current')
assert any(r['returncode']!=0 and 'format-overflow' in r['stderr'] for r in results if r['label']=='old')
