from pathlib import Path
import subprocess,json
p=Path(__file__).parent;base=json.loads((p/'budget-commands.json').read_text())[-2]['command'];oldsource=next(x for x in base if x.endswith('/old-budget.c'));oldexe=base[-1];rows=[]
# Use the real NanoISA query closure, not the older full-VM budget harness closure.
import shlex
agg=Path('/Users/jkh/Src/nanolang-51-harness')
parts=dict(line.split('=',1) for line in Path('/private/tmp/nanolang-map-paired-publication-20261022/consumers.txt').read_text().splitlines())
objects=[str(agg/x) for x in shlex.split(parts['nanoisa']) if Path(x).stem not in ['dump_main','managed_array_shapes']]
for stem in ['verifier','ownership_contracts','nvm_v2_layouts']:
 objects=[str(p/('budget-'+stem+'-alloc.o')) if Path(x).stem==stem else x for x in objects]
for stem in ['verifier_types','vm_decode','retained_layouts']:
 wrapped=next(x for x in base if Path(x).stem==stem and x.endswith('.o'))
 objects=[wrapped if Path(x).stem==stem else x for x in objects]
base=base[:base.index(oldsource)]+[oldsource,'tests/nanoisa/record_array_alloc.c',*objects,'-lm','-lcrypto','-lffi','-L/opt/homebrew/Cellar/openssl@4/4.0.3/lib','-o',oldexe]

for case in ['origins','execution','image']:
 name='neighbor-map-'+case;exe=p/name;args=[('tests/nanoisa/test_record_map_'+case+'.c') if x==oldsource else str(exe) if x==oldexe else x for x in base]
 for name,args in [(name+'-build',args),(name+'-run',[str(exe)])]:
  r=subprocess.run(args,capture_output=True,timeout=180);(p/(name+'.stdout')).write_bytes(r.stdout);(p/(name+'.stderr')).write_bytes(r.stderr);rows.append({'name':name,'argv':args,'status':r.returncode});(p/'query-neighbors.json').write_text(json.dumps(rows,indent=2));assert r.returncode==0 and not r.stderr,(name,r.returncode,r.stderr);print(r.stdout.decode(),end='')
print('PASS three old map origin/recipe/image neighbors with current included proof and rebuilt header dependents')
