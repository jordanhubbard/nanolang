from pathlib import Path
import subprocess,json
root=Path('/private/tmp/nanolang-match-guards-20261009');out=Path('/private/tmp/nanolang-introspection-owned-c-20261009')
s=Path('/private/tmp/nanolang-introspection-routes-20261009/owned_direct.nano').read_text()
cases={'leak':s.replace('assert (== (consume h) 7)',''), 'moved':s.replace('assert (== (consume h) 7)','assert (== (consume h) 7) assert (== (consume h) 7)'), 'signature':s+'\nextern fn ___module_name_route_probe() -> int\n'}
for name,text in cases.items():
 source=out/(name+'.nano');source.write_text(text);target=out/(name+'.nvm');target.write_bytes(b'previous artifact')
 r=subprocess.run([str(out/'nano_virt'),str(source),'--emit-nvm','-o',str(target)],cwd=root,capture_output=True,text=True,timeout=120)
 preserved=target.read_bytes()==b'previous artifact';print(json.dumps({'case':name,'exit':r.returncode,'preserved':preserved,'stdout':r.stdout,'stderr':r.stderr}),flush=True)
 assert r.returncode and preserved
