import sys,json,pathlib,hashlib,shutil,os
sys.dont_write_bytecode=True
root=pathlib.Path(sys.argv[1]).resolve();previous=pathlib.Path(sys.argv[2]).resolve();report=pathlib.Path(sys.argv[3]).resolve();fixture=pathlib.Path(sys.argv[4]).resolve()
sys.path.insert(0,str(root))
from tests.test_portable_read_adapters import PortableReadAdapters
x=PortableReadAdapters();x.artifacts=report;report.mkdir(exist_ok=False);x.store=report/'objects';x.store.mkdir();x.work=report/'products';x.work.mkdir();x.index=0
config=json.loads((previous/'configuration.json').read_text());baseline=json.loads((previous/'inputs-after.json').read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for path,row in baseline.items():assert sha(path)==row['sha256'],path
last=sorted(previous.glob('*-products-after.json'))[-1];products=json.loads(last.read_text())
objects=[previous/'products'/f'unhooked-{i}.o' for i in range(3)];llvm={n:previous/'products'/f'typed-{n}.o' for n in ['O0','O2']}
for p in [*objects,*llvm.values()]:assert sha(p)==products[p.name]['sha256']
x.inputs=sorted(set([pathlib.Path(p) for p in baseline]+[fixture,*objects,*llvm.values(),pathlib.Path(__file__).resolve()]),key=str)
x.env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='')
if 'SDKROOT' not in x.env:raise RuntimeError('explicit SDKROOT required')
x.dump('reuse.json',{'original_products_map':str(last),'original_config':config,'fixture':str(fixture),'fixture_sha256':sha(fixture),'baseline_files_verified':len(baseline),'objects_verified':{str(p):sha(p) for p in [*objects,*llvm.values()]},'scope':'Only previously unrun Homebrew ordinary unhooked C/O0/O2 routes; original observed passes unchanged.'})
shutil.copyfile(__file__,report/'driver.py');before=x.inventory();x.dump('inputs-before.json',before)
try:
 common=[*config['cc'],'-std=c11','-D_POSIX_C_SOURCE=200809L','-O1','-g','-Wall','-Wextra','-Werror','-Isrc/nanoisa',*config['flags']]
 for route in ['C','O0','O2']:
  binary=x.work/('unhooked-'+route);flags=[] if route=='C' else ['-DREAD_LLVM'];link=[] if route=='C' else [llvm[route]]
  x.command([*common,*flags,fixture,*objects,*link,'-lm','-o',binary])
  files=x.work/('unhooked-'+route+'-files');files.mkdir();output=x.command([binary,files]);assert 'private read-text adapter checks; no bytecode admission.' in output;print(output,end='',flush=True)
finally:
 after=x.inventory();x.dump('inputs-after.json',after);x.dump('phase-summary.json',{'commands':x.index,'inputs_equal':before==after});assert before==after
