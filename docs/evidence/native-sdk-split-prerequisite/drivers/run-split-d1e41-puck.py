import pathlib,json,hashlib,os,sys,tarfile
root=pathlib.Path('/tmp/nanolang-native-sdk-d1e41-puck-source');report=pathlib.Path('/tmp/nanolang-split-d1e41-puck-evidence');old=pathlib.Path('/tmp/nanolang-native-sdk-1ee9-puck-source');prior=pathlib.Path('/tmp/nanolang-split-1ee9-puck-evidence')
assert not root.exists() and not report.exists()
root.mkdir();report.mkdir()
with tarfile.open('/tmp/native-sdk-d1e41-source.tar.gz') as archive:
 for m in archive.getmembers():assert m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
 archive.extractall(root)
sys.path.insert(0,str(root));from tests.native_sdk_runner import run
sel=json.loads(pathlib.Path('/tmp/native-sdk-d1e41-selected-git.json').read_text())
def ident(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return dict(sha256=h.hexdigest(),bytes=p.stat().st_size,mode=p.stat().st_mode&0o7777)
def dump(n,v):(report/n).write_text(json.dumps(v,indent=2)+'\n')
def check(base,expected):
 actual={p:ident(base/p) for p in expected};assert actual==expected;return actual
sources=json.loads((prior/'source-after.json').read_text());products=json.loads((prior/'products-after.json').read_text());tools=json.loads((prior/'tools-after.json').read_text())
fixture={r['path']:dict(sha256=r['sha256'],bytes=r['bytes'],mode=int(r['mode'],8)&0o7777) for r in sel['records']}
manifest=json.loads((prior/'producers.json').read_text())
for role in ('refresh1','refresh2'):
 status=json.loads((prior/(role+'-status.json')).read_text());assert status['returncode']==0,status
 row=manifest['producers'][role];assert ident(row['path'])=={k:row[k] for k in ('sha256','bytes','mode')}
manifest.update(selected_roles=['refresh1','refresh2'],fixture_pin=sel['pin'],scope='Two retained exact 1ee9 Nano refresh products and original SDK providers; d1 primitive fixture only, not fresh bootstrap or complete SDK corpus')
manifest['producers']={k:manifest['producers'][k] for k in ('refresh1','refresh2')};dump('producers.json',manifest)
for label,base,expected in [('source',old,sources),('products',old,products),('tools',pathlib.Path('/'),tools),('fixture',root,fixture)]:dump(label+'-before.json',check(base,expected))
dump('helper-identity-proof.json',json.loads(pathlib.Path('/tmp/split-d1-reviewed-helper-comparison.json').read_text()))
tmp=report/'tmp';tmp.mkdir();env=dict(TMPDIR=str(tmp),NANOLANG_SDK_ROOT=str(old),NANO_SPLIT_REPORT_DIR=str(report),NANO_SPLIT_PRODUCER_MANIFEST=str(report/'producers.json'),NANO_SHADOW_TIMING=None,NANO_SHADOW_TIMEOUT_SECONDS=None,NANO_SDK_MIN_FREE_BYTES=str(2*1024**3),PATH='/opt/homebrew/bin:'+os.environ['PATH'],CC='/usr/bin/clang',NANO_CC='/usr/bin/clang',PKG_CONFIG_PATH='/opt/homebrew/opt/libffi/lib/pkgconfig:/opt/homebrew/opt/openssl@3/lib/pkgconfig',LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib',DYLD_LIBRARY_PATH='/opt/homebrew/opt/openssl@3/lib')
first=None
try:
 out,_,_=run(report,'sdk-path',['/usr/bin/xcrun','--show-sdk-path'],root,env,track_descendants=True);env['SDKROOT']=out.decode().strip();dump('configuration.json',dict(environment=env,fixture_pin=sel['pin'],retained_pin=manifest['source_pin']))
 run(report,'split-primitive',[sys.executable,'-m','unittest','-f','-v','tests.test_str_split_primitive'],root,env,timeout=1800,track_descendants=True)
except BaseException as e:first=repr(e);dump('first-failure.json',dict(error=first));raise
finally:
 for label,base,expected in [('source',old,sources),('products',old,products),('tools',pathlib.Path('/'),tools),('fixture',root,fixture)]:dump(label+'-after.json',check(base,expected))
 dump('terminal.json',dict(first_failure=first,identities_equal=True))
print('PASS',flush=True)
