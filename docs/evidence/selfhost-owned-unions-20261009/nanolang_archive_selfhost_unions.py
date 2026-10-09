from pathlib import Path
import hashlib,json,shutil,subprocess
checkout=Path('/private/tmp/nanolang-match-guards-20261009')
source=Path('/private/tmp/nanolang-selfhost-union-final')
report=json.loads((source/'manifest.json').read_text())
assert len(report['steps'])==5 and [s['exit'] for s in report['steps']]==[0,0,0,0,1]
assert report['unchanged']
supplement=Path('/private/tmp/nanolang-selfhost-union-supplement')
corrected=json.loads((supplement/'manifest.json').read_text())
assert len(corrected['steps'])==6 and all(s['exit']==0 for s in corrected['steps'])
assert all(corrected[key] for key in ('unchanged','bootstrap_consumers_unchanged','nano_sources_unchanged'))
fresh = json.loads(Path('/private/tmp/nanolang-selfhost-union-fresh-installed/manifest.json').read_text())
assert fresh['exit'] == 0 and fresh['unchanged']
for p,digest in corrected['after'].items():
    assert hashlib.sha256((checkout/p).read_bytes()).hexdigest()==digest,p
out=checkout/'docs/evidence/selfhost-owned-unions-20261009'
out.mkdir(parents=True,exist_ok=True)
def archive_tree(origin,destination):
    inventory=[]
    for p in sorted(origin.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(origin);data=p.read_bytes()
        keep=p.suffix in ('.nano','.nasm','.c','.json','.log','.patch','.py','.md')
        inventory.append({'path':str(rel),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'text_archived':keep})
        if keep:
            target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'artifact-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
archive_tree(source,out/'qualification')
archive_tree(supplement,out/'supplement')
archive_tree(Path('/private/tmp/nanolang-selfhost-union-seed-cache'),out/'seed-cache-comparison')
archive_tree(Path('/private/tmp/nanolang-selfhost-union-fresh-installed'),out/'fresh-installed')
archive_tree(Path('/private/tmp/nanolang-ci-gcc-evidence'),out/'gcc-repair')
archive_tree(Path('/private/tmp/nanolang-selfhost-union-initial-evidence'),out/'initial')
archive_tree(Path('/private/tmp/nanolang-selfhost-union-boundaries'),out/'initial-boundaries')
boot=json.loads((checkout/'bin/nanoc_bootstrap.json').read_text())
assert boot['raw_stage1_stage2_equal']
assert all(s.get('exit_code') == 0 for s in boot['steps'])
archive_tree(Path(boot['work']),out/'fresh-bootstrap')
original=json.loads(Path('/private/tmp/nanolang-selfhost-union-seed-cache/original-bootstrap.json').read_text())
archive_tree(Path(original['work']),out/'bootstrap')
for filename in ('nanolang-selfhost-union-nested-build.log','nanolang-selfhost-union-final-driver.log','nanolang_selfhost_union_final_gate.py','nanolang_selfhost_union_supplement.py','nanolang-selfhost-union-supplement-driver.log'):
    shutil.copy2(Path('/private/tmp')/filename,out/filename)
(out/'source.patch').write_text(subprocess.check_output(['git','diff','--','src/nanovirt/borrow_codegen.inc','src_nano/compiler/nanoisa_borrows.nano'],cwd=checkout,text=True))
(out/'tools.json').write_text(json.dumps({p:hashlib.sha256((checkout/p).read_bytes()).hexdigest() for p in ['bin/nanoc_c','bin/nano_virt','bin/nanoc_stage1','bin/nanoc_stage2','bin/nanoisa_emit','bin/nanoisa','bin/nano_vm','bin/nvm2c']},indent=2)+'\n')
print(out)
