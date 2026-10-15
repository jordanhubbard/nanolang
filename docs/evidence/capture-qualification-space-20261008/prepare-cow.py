import ctypes,json,os,subprocess,time
from pathlib import Path
source=Path('/Users/jordanh/Src/nanolang');head='ae92c0488ed44da0112f7eb51367a9976054b16a'
root=Path('/private/tmp/nanolang-capture-ae92c0488-cow')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==head
assert not subprocess.check_output(['git','diff','HEAD','--'],cwd=source)
subprocess.run(['git','clone','--shared','--no-checkout',str(source),str(root)],check=True)
lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
lib.clonefile.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int];lib.clonefile.restype=ctypes.c_int
paths=subprocess.check_output(['git','ls-tree','-r','--name-only','-z',head],cwd=source).split(b'\0')
count=0;start=time.monotonic()
for raw in paths:
 if not raw:continue
 rel=os.fsdecode(raw);a=source/rel;b=root/rel;b.parent.mkdir(parents=True,exist_ok=True)
 if a.is_symlink():os.symlink(os.readlink(a),b)
 else:
  if lib.clonefile(os.fsencode(a),os.fsencode(b),0):raise OSError(ctypes.get_errno(),rel)
 count+=1
subprocess.run(['git','read-tree',head],cwd=root,check=True)
subprocess.run(['git','update-ref','--no-deref','HEAD',head],cwd=root,check=True)
status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
assert not status,status
record={'source_commit':head,'method':'APFS clonefile of every tracked path from clean committed source plus shared object database; independent index; clean git status verified','files':count,'seconds':time.monotonic()-start,'source':str(source),'destination':str(root),'status':status}
Path('/private/tmp/nanolang-capture-cow-checkout.json').write_text(json.dumps(record,indent=2)+'\n')
s=Path('/private/tmp/nanolang-closure-arrays-9790d9a91-run.py').read_text().replace('nanolang-closure-arrays-9790d9a91','nanolang-capture-ae92c0488-cow').replace('9790d9a91aac6b412e48e2f6af04c8f5a6392be1',head)
Path('/private/tmp/nanolang-capture-current-run.py').write_text(s)
print(json.dumps(record),flush=True)
