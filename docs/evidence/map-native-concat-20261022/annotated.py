from pathlib import Path
import json,sys,hashlib
sys.path.insert(0,str(Path.cwd()))
from tests.native_process import run_captured
root=Path.cwd();agg=Path('/Users/jkh/Src/nanolang-51-harness');out=Path(__file__).parent
w=Path('/var/folders/v0/pxyl2_m512v5j9nhv674pt400000gn/T/nano-record-map-public-8d6uwbpz');s=json.loads((w/'selection.json').read_text())
source=(root/'tests/nanoisa/test_record_map_public.c').read_text();start=source.index('static void public_original_module(');end=source.index('\nstatic void borrowed_integer',start);fn=source[start:end].replace('public_original_module','concat_original_module').replace('public_check(m,true);','public_check(m,native_expected); if(!native_expected){char error[512]; char *native=nvm2c_emit(m,error,sizeof error); CHECK(!native); CHECK(strstr(error,"string producer")); printf("Expected selected native refusal: %s\\n",error);}')
text='#define MAP_PUBLIC_MAIN retained_public_main\n#include "'+str(root/'tests/nanoisa/test_record_map_public.c')+'"\nstatic bool native_expected;\n'+fn+'\nint main(int argc,char **argv){CHECK(argc==9);for(int i=1;i<argc;i++){native_expected=i<=4;concat_original_module(argv[i]);}printf("PASS %u annotated baseline checks\\n",checks);return 0;}\n'
(out/'annotated.c').write_text(text)
objects=[str(w/('public-switch-'+Path(x).stem+'.o')) for x in s['sources']]+[str(agg/x) if not Path(x).is_absolute() else x for x in s['objects']]
cmd=s['cc']+s['flags']+['-DNANO_NO_COMPUTED_GOTO',str(out/'annotated.c')]+objects+s['ldflags']+['-o',str(out/'annotated')]
rows=[];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pins={p:sha(p) for p in objects}
for name,args in [('build',cmd),('run',[str(out/'annotated')]+[str(out/(n+'-'+p+'.nvm')) for n in ['unchanged','concat'] for p in ['seed','c-hosted','stage1','stage2']])]:
 r=run_captured(args,cwd=agg,env={'RECORD_MAP_PUBLIC_OUTPUT':str(out)},timeout=180,build=name=='build');(out/('annotated-'+name+'.stdout')).write_bytes(r.stdout);(out/('annotated-'+name+'.stderr')).write_bytes(r.stderr);rows.append(dict(name=name,argv=args,status=r.returncode));(out/'annotated-status.json').write_text(json.dumps(rows,indent=2)+'\n');assert r.returncode==0 and not r.stderr,r.stderr
assert pins=={p:sha(p) for p in objects};(out/'annotated-provider-pins.json').write_text(json.dumps(pins,indent=2)+'\n');print((out/'annotated-run.stdout').read_text())
