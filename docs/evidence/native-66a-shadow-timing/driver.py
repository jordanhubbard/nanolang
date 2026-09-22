from pathlib import Path
import hashlib,json,os,shutil,subprocess,importlib.util
old=Path('/home/jkh/nanolang-qualification/record-lists-66a-linux-source')
prep=old.parent/'record-lists-66a-linux-prepare'
root=old.parent/'record-lists-66a-observer-source'
report=old.parent/'record-lists-66a-observer-report'
observer=Path('/home/jkh/Src/nanolang-native-66a-shadow-timing')
root.mkdir();report.mkdir()
def ident(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return dict(sha256=h.hexdigest(),bytes=p.stat().st_size,mode=p.stat().st_mode&0o7777)
def dump(n,v):(report/(n+'.json')).write_text(json.dumps(v,indent=2)+'\n')
maps={kind:json.loads((prep/('bootstrap-'+kind+'-after.json')).read_text()) for kind in ('sources','products','tools')}
for kind,entries in maps.items():
 for p,v in entries.items():
  a=ident(Path(p));assert a['sha256']==v['sha256'] and a['bytes']==v['bytes'],(kind,p)
assert shutil.disk_usage(root).free>2*1024**3+sum(v['bytes'] for k in ('sources','products') for v in maps[k].values())
for kind in ('sources','products'):
 for p in maps[kind]:
  p=Path(p);dst=root/p.relative_to(old);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
oldident={kind:{p:ident(Path(p)) for p in entries} for kind,entries in maps.items()}
for kind in ('sources','products'):
 assert all(ident(root/Path(p).relative_to(old))==v for p,v in oldident[kind].items())
changed=['Makefile.gnu','src/main.c','src/eval.c','src/runtime/shadow_timing.h']
for name in changed:shutil.copy2(observer/name,root/name)
sources={str(p.relative_to(root)):ident(p) for p in root.rglob('*') if p.is_file() and (str(old/p.relative_to(root)) in maps['sources'] or str(p.relative_to(root)) in changed)}
runner=Path('/tmp/nanolang-native-sdk-fe645-linux-diagnostic-source/tests/native_sdk_runner.py');shutil.copy2(runner,report/'native_sdk_runner.py')
spec=importlib.util.spec_from_file_location('native_timing_runner',report/'native_sdk_runner.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
private=report/'tmp';private.mkdir()
env=json.loads((prep/'environment.json').read_text());env.update(NANOLANG_SDK_ROOT=str(root),NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_MODULE_PATH=str(root/'modules'),TMPDIR=str(private),BOOTSTRAP_TMPDIR=str(private),NANO_CC='/bin/gcc',NANO_SHADOW_TIMING='1',NANO_SHADOW_TIMEOUT_SECONDS=None,NANO_SDK_MIN_FREE_BYTES=str(2*1024**3))
dump('environment',env);dump('source-before',sources);dump('frozen-endpoints',oldident)
dump('selection',dict(production='66a11b6dbf1cf1bef56698f0e1fcbe516276c270',observer=subprocess.check_output(['git','rev-parse','HEAD'],cwd=observer,text=True).strip(),changed=changed,runner=ident(runner),script=ident(Path(__file__)),load=os.getloadavg(),cpu_count=os.cpu_count(),scope='single observer diagnostic; only main/eval rebuilt; no supervisor correction; original child10/outer120'))
fragment=report/'observer.mk';fragment.write_text('include Makefile.gnu\n.PHONY: observer-build\nobserver-build:\n\t$(CC) $(CFLAGS) -c src/main.c -o obj/main.o\n\t$(CC) $(CFLAGS) -c src/eval.c -o obj/eval.o\n\t$(CC) $(CFLAGS) -o $(COMPILER_C) $(COMPILER_OBJECTS) $(LDFLAGS)\n')
first=None
try:
 module.run(report,'observer-build',['make','-f',fragment,'observer-build','CC=/bin/gcc'],root,env,timeout=180,track_descendants=True)
 replaced={'obj/main.o','obj/eval.o','bin/nanoc_c'}
 assert all(ident(root/Path(p).relative_to(old))==v for p,v in oldident['products'].items() if str(Path(p).relative_to(old)) not in replaced)
 dump('replaced-products',{p:ident(root/p) for p in replaced});dump('provider-proof',dict(unchanged_count=sum(str(Path(p).relative_to(old)) not in replaced for p in oldident['products']),replaced=sorted(replaced)))
 output=report/'stage1';sentinel=b'diagnostic output sentinel\n';output.write_bytes(sentinel)
 _,_,status=module.run(report,'timed-bootstrap',[root/'bin/nanoc_c','src_nano/nanoc_v06.nano','-o',output,'--verbose','--llm-shadow-json',report/'shadows.json'],root,env,expected=(0,1),timeout=120,track_descendants=True)
 if status['returncode']!=0:assert output.read_bytes()==sentinel
 dump('output',dict(identity=ident(output),sentinel_preserved=output.read_bytes()==sentinel))
except BaseException as e:first=repr(e);dump('first-failure',dict(error=first));raise
finally:
 final={p:ident(root/p) for p in sources};dump('source-after',final)
 unchanged=all(ident(Path(p))==v for entries in oldident.values() for p,v in entries.items())
 dump('terminal',dict(first_failure=first,source_equal=final==sources,frozen_sources_tools_products_equal=unchanged,load=os.getloadavg()))
 assert final==sources and unchanged
