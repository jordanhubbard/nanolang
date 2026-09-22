import pathlib,subprocess,json,hashlib,sys,shlex,os
repo=pathlib.Path('/home/jkh/Src/nanolang-native-sdk-integration')
root=pathlib.Path('/tmp/nanolang-native-sdk-dc9-linux-direct-source');root.mkdir()
report=pathlib.Path('/tmp/nanolang-native-sdk-dc9-linux-direct');report.mkdir()
old=pathlib.Path('/tmp/nanolang-native-sdk-bae59-linux-source');prep=pathlib.Path('/tmp/nanolang-native-sdk-bae59-linux-preparation')
pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert pin=='dc9c37fc177dee5c71e6bfc6f1efb0a2bcb7e87c'
names=subprocess.check_output(['git','ls-tree','--name-only',pin],cwd=repo,text=True).splitlines();names=[n for n in names if n!='docs']
archive=subprocess.Popen(['git','archive',pin,*names],cwd=repo,stdout=subprocess.PIPE)
subprocess.run(['tar','-xf','-','-C',str(root)],stdin=archive.stdout,check=True);archive.stdout.close();assert archive.wait()==0
sys.path.insert(0,str(root));from tests.native_sdk_runner import run

def ident(p):
 p=pathlib.Path(p);return dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,mode=p.stat().st_mode&0o7777)
def dump(n,v):(report/n).write_text(json.dumps(v,indent=2)+'\n')
oldsource=json.loads((prep/'source-after.json').read_text());oldproducts=json.loads((prep/'products-after.json').read_text());oldtools=json.loads((prep/'tools-after.json').read_text())
assert all(ident(p)==v for p,v in oldtools.items())
closure={p:v for p,v in oldsource.items() if p.startswith(('src/','modules/','schema/','scripts/')) or p.startswith('Makefile')}
changed=[p for p,v in closure.items() if ident(root/p)!=v];assert changed==['src/typechecker.c'],changed
assert all(ident(old/p)==v for p,v in closure.items())
assert all(ident(old/p)==v for p,v in oldproducts.items())
tracked=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
def sources():return {str(p.relative_to(root)):ident(p) for p in tracked}
before=sources();dump('source-before.json',before);dump('tools-before.json',oldtools)
dump('reuse-proof.json',dict(source_pin=pin,provider_pin='bae599477588b57b89159e10f03d5f5518ca6093',provider_preparation=str(prep),retained_original_terminal=json.loads((prep/'terminal.json').read_text()),c_closure=closure,changed_c_inputs=changed,excluded_objects=['main.o','typechecker.o'],scope='fresh included checker and exact retained ordinary providers; no imports, module compilation, snapshot extraction, producer refresh or bootstrap'))
private=report/'tmp';private.mkdir()
env=dict(TMPDIR=str(private),CC='/usr/bin/gcc',NANO_CC='/usr/bin/gcc',NANOLANG_SDK_ROOT=str(root),NANO_SDK_MIN_FREE_BYTES=str(2*1024**3),NANO_SHADOW_TIMING=None)
first=None
try:
 query=report/'query.mk';query.write_text('split_inventory:;@printf "%s\\n" "SPLIT_OBJECTS=$(COMPILER_OBJECTS)" "SPLIT_CFLAGS=$(CFLAGS)" "SPLIT_LDFLAGS=$(LDFLAGS)"\n')
 out,_,_=run(report,'provider-query',['make','-f','Makefile.gnu','-f',query,'-s','split_inventory'],root,env,timeout=180,track_descendants=True)
 fields=dict(line.split('=',1) for line in out.decode().splitlines() if line.startswith('SPLIT_'))
 objects=[old/p for p in shlex.split(fields['SPLIT_OBJECTS']) if pathlib.Path(p).name not in ('main.o','typechecker.o')]
 assert objects and all(ident(p)==oldproducts[str(p.relative_to(old))] for p in objects)
 dump('providers-before.json',{str(p):ident(p) for p in objects})
 env.update(NANO_SPLIT_CC='/usr/bin/gcc',NANO_SPLIT_CFLAGS=fields['SPLIT_CFLAGS'],NANO_SPLIT_LDFLAGS=fields['SPLIT_LDFLAGS'],NANO_SPLIT_CONTEXT_OBJECTS=shlex.join(str(p) for p in objects),NANO_SPLIT_REPORT_DIR=str(report))
 run(report,'direct-context',[sys.executable,'-m','unittest','-f','-v','tests.test_str_split_result_context'],root,env,timeout=900,track_descendants=True)
except BaseException as e:
 first=repr(e);dump('first-failure.json',dict(error=first));raise
finally:
 after=sources();tools={p:ident(p) for p in oldtools};dump('source-after.json',after);dump('tools-after.json',tools)
 product_equal=all(ident(old/p)==v for p,v in oldproducts.items());dump('terminal.json',dict(first_failure=first,source_equal=before==after,tools_equal=tools==oldtools,retained_products_equal=product_equal));assert before==after and tools==oldtools and product_equal
print('PASS',flush=True)
