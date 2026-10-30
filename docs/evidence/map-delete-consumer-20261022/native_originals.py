from pathlib import Path
import subprocess,json,hashlib
p=Path(__file__).parent
s=json.loads((p/'focused.json').read_text());cc=s['retained_selection']['cc'];rows=[]
def run(name,args):
 r=subprocess.run(list(map(str,args)),capture_output=True,timeout=180);(p/(name+'.stdout')).write_bytes(r.stdout);(p/(name+'.stderr')).write_bytes(r.stderr);rows.append({'name':name,'args':list(map(str,args)),'status':r.returncode});(p/'native-originals.json').write_text(json.dumps(rows,indent=2));assert r.returncode==0 and not r.stdout and not r.stderr,(name,r.returncode,r.stdout,r.stderr)
for name in ['seed','canonical','c-shadow','raw-shadow']:
 text=(p/(name+'.c')).read_text();needle='if (debt >= nmap_collection_budget) nmap_collect();';assert text.count(needle)==1
 clear='for(size_t i=0;i<sizeof nglobal/sizeof nglobal[0];i++)nglobal[i]=(nmap_value){0};' if 'static nmap_value nglobal[' in text else ''
 forced='#define main fixture_main\n'+text.replace(needle,'(void)debt; nmap_collect();')+'\n#undef main\nint main(void){int r=fixture_main();'+clear+'nmap_collect();if(nmap_owned_head||nrec_owned_head||nmap_live_bytes||nagg_live_bytes||nroot_head)abort();return r;}\n'
 for mode,body in [('raw',text),('forced',forced)]:
  source=p/(name+'-'+mode+'.c');source.write_text(body)
  for opt in ['0','2']:
   tag=name+'-'+mode+'-O'+opt;exe=p/tag
   run(tag+'-build',cc+['-std=c11','-g','-O'+opt,'-fsanitize=address,undefined','-fno-sanitize-recover=all',source,'-lm','-o',exe]);run(tag+'-run',[exe])
print('PASS 16 original DELETE native executions O0/O2 ASan/UBSan, raw and forced collection')
