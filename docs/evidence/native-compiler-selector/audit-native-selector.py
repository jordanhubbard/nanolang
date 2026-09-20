import json,hashlib,subprocess
from pathlib import Path
base=Path('/tmp/nanolang-native-selector-independent');repo='/home/jkh/Src/nanolang-capture-flags'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
results={'production_pin':'3ef5d2261826a55941e399a155cb5bb178b821c3','runs':[],'archives':{},'status':'PASS_WITH_SCOPE_LIMITS'}
for k in ['diagnostic','selected']:
 p=Path('/tmp/nanolang-darwin-native-'+k+'-evidence.tar.gz');results['archives'][k]={'sha256':sha(p),'bytes':p.stat().st_size}
roots=[Path('/tmp/nanolang-linux-'+n+'-selected') for n in ['projected','tuple']]+list((base/'selected').iterdir())+[base/'diagnostic/nanolang-darwin-projected-diagnosis',base/'diagnostic/nanolang-darwin-asan-startup-control']
for root in roots:
 if not root.is_dir():continue
 manifest=root/('manifest.json' if root.name.endswith('startup-control') else 'products.json');m=read(manifest)
 for path,v in m.items():
  parts=Path(path).parts;i=parts.index(root.name);p=root.joinpath(*parts[i+1:]);assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],p
 pairs=[]
 for stem in ['inputs','providers']:
  p=root/(stem+'-before.json')
  if p.exists():a=read(p);assert a==read(root/(stem+'-after.json'));pairs.append({'kind':stem,'count':len(a)})
 statuses={p.name:read(p) for p in root.glob('*-status.json')};fail={k:v for k,v in statuses.items() if v.get('returncode') or v.get('timeout')}
 selected='selected' in root.name
 compiles=[];runs=[]
 if selected:
  assert read(root/'result.json')=={'tests':1,'errors':0,'failures':0,'passed':True};assert not fail
  for p in root.glob('*-command.json'):
   x=read(p);argv=x['argv']
   if '-fsanitize=address,undefined' in argv:
    assert all(f in argv for f in ['-std=c11','-O1','-g','-fno-omit-frame-pointer','-Wall','-Wextra','-Werror'])
    assert argv[0]==('/opt/homebrew/opt/llvm/bin/clang' if 'darwin' in root.name else 'cc')
    if 'tuple' in root.name:assert '-fno-sanitize-recover=all' in argv
    compiles.append(argv);binary=argv[argv.index('-o')+1]
    execution=[(q,read(q)) for q in root.glob('*-command.json') if read(q)['argv']==[binary]];assert len(execution)==1
    q,execdata=execution[0];assert execdata['timeout']==120
    status=read(root/q.name.replace('command','status'));assert status['returncode']==0 and not status['timeout'];assert status.get('leader_reaped') and status['group_disappeared'];runs.append({'binary':binary,'status':status,'asan_options':execdata.get('selected_ASAN_OPTIONS')})
  assert len(compiles)==len(runs)==2
 results['runs'].append({'root':str(root),'manifest_files_verified':len(m),'equal_pairs':pairs,'commands':len(statuses),'failures':fail,'selected_native_runs':runs})
# Exact source transformation; all other bytes/assertions/deadlines are preserved.
source=[]
for f in ['tests/test_nanoisa_flat_records.py','tests/test_nanoisa_tuple_values.py']:
 before=subprocess.check_output(['git','-C',repo,'show','3ef5d2261^:'+f]).decode();after=subprocess.check_output(['git','-C',repo,'show','3ef5d2261:'+f]).decode()
 restored=after.replace('import shlex\n','').replace('*shlex.split(os.environ.get("NANO_NATIVE_TEST_CC", "cc")),\n                        "-std=c11",','"cc", "-std=c11",') if 'flat_records' in f else after.replace('import shlex\n','').replace('*shlex.split(os.environ.get("NANO_NATIVE_TEST_CC", "cc")),','"cc",')
 assert restored==before,f
 source.append({'path':f,'sha256':hashlib.sha256(after.encode()).hexdigest(),'selector_only_change':True})
 # Each selected input map contains the exact reviewed fixture bytes.
 for run in results['runs']:
  if 'selected' not in run['root']:continue
  inp=read(Path(run['root'])/'inputs-before.json')
  if ('projected' in run['root'])==('flat_records' in f):assert any(Path(p).name==Path(f).name and v['sha256']==source[-1]['sha256'] for p,v in inp.items())
results['source']=source
sample=(base/'diagnostic/nanolang-darwin-projected-diagnosis/store-first-sample.txt').read_text();assert '__asan::AsanInitInternal()' in sample and '__asan::InitializeShadowMemory()' in sample
control=base/'diagnostic/nanolang-darwin-asan-startup-control';assert read(control/'apple-run-status.json')['timeout'];assert read(control/'homebrew-run-status.json')['returncode']==0
results['diagnosis']={'live_sample_in_apple_asan_initialization':True,'minimal_apple_timeout_seconds':10,'minimal_homebrew_stdout':(control/'homebrew-run.stdout').read_text(),'projected_timeout_seconds_each':120}
results['limits']=['Four selected methods only, not the whole 90-method suite.','Historical deleted binaries remain unattributed; fresh reproducible Apple startup diagnostics do not retroactively prove every prior timeout cause.','Tuple Darwin retains its existing detect_leaks=0 policy; no Darwin tuple LSan acceptance inferred.','Frozen compiler/provider selection, not current canonical bootstrap or all transitive toolchain files.','Read-only artifact audit; no test reruns.']
results['audit_script_sha256']=sha('/tmp/audit-native-selector.py')
p=Path('/tmp/nanolang-native-selector-independent-audit.json');p.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n');print(sha(p));print([(x['root'],x['manifest_files_verified'],len(x['failures'])) for x in results['runs']])
