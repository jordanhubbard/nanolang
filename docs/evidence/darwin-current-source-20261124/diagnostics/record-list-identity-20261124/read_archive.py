"""I verify owned-record list red/green without accepting broader source gates."""
from pathlib import Path
import sys,json,subprocess,hashlib
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent));from read_chunks import read_archive
m,rows,data,js=read_archive(base,'404856b441a461ed6e711fe01a85a865538786868951253d24be17652e396905')
sha=lambda raw:hashlib.sha256(raw).hexdigest();assert m['source_pin']=='97bb958fd062f6a76db6e938a89f73b38fb0304b' and m['baseline_pin']=='5bf50072752f6e88bad274058663e6e7effc1f89'
assert not m['full_source_accepted'] and not m['whole_release']
red=js('journal/result.json');v1=js('journal/qualified-result.json');failed=js('journal/profiles-result.json');final=js('journal/profiles-v2-result.json');neighbors=js('journal/v2-neighbors-result.json')
assert red['terminal'] and red['commands'][-1]['label']=='red' and red['commands'][-1]['exit']==1
assert b'unknown type name' in data('journal/red.stderr') and b'List_Item' in data('journal/red.stderr')
assert rows['journal/baseline-nanoc_c']['sha256']==red['baseline_image']
assert v1['accepted'] and v1['terminal'] and all(c['exit']==0 for c in v1['commands'])
assert failed['terminal'] and not failed.get('accepted') and len(failed['profiles'])==1
assert failed['profiles'][0]['mode']=='san' and failed['profiles'][0]['commands'][-1]['exit']==1
assert b'LeakSanitizer' in data('journal/san-record-tests.stderr') and b'FAILED (failures=2)' in data('journal/san-record-tests.stderr')
assert final['accepted'] and final['terminal'] and [p['mode'] for p in final['profiles']]==['normal','san','production']
assert final['source_sha256']==rows['journal/typechecker-v2.c']['sha256']
assert b'detect_leaks=1:halt_on_error=1' in data('journal/profiles-v2.py')
assert b'UBSAN_OPTIONS=\'halt_on_error=1\'' in data('journal/profiles-v2.py')
prior=base.parent/'list-source-linux-20261124';pm=json.loads((prior/'manifest.json').read_text());pr={r['path']:r for r in pm['files']}
assert pr['normal/bin/nanoc_c']['sha256']==red['baseline_image']
for p in final['profiles']:
    mode=p['mode'];assert all(c['exit']==0 for c in p['commands']) and p['providers']==p['providers_after']
    assert rows['journal/source/bin/nanoc_record_v2_'+mode]['sha256']==p['image_sha256']
    for name,h in p['providers'].items():assert pr[mode+'/'+name]['sha256']==h
    original=p['original_link_args'];linked=p['commands'][1]['args'];i=original.index('-o')
    expected=list(original);expected[i+1]=linked[i+1];expected=[p['commands'][0]['args'][-1] if a=='obj/typechecker.o' else a for a in expected];assert expected==linked
    assert final['test_sha256']==rows['journal/source/tests/test_cseed_record_lists.py']['sha256']
    assert b'Ran 6 tests' in data('journal/v2-'+mode+'-record-tests.stderr') and data('journal/v2-'+mode+'-record-tests.stderr').endswith(b'OK\n')
    reports=[n for n in rows if n.startswith('journal/v2-'+mode+'-cases/') and n.endswith('/result.json')];assert len(reports)==6
    rejected=0
    for name in reports:
        r=js(name);directory=name.rsplit('/',1)[0];assert r['compiler_sha256']==p['image_sha256'] and r['shadow_timeout']=='default10'
        if r['reject']:
            rejected+=1;assert r['exit']!=0 and data(directory+'/program')==b'I retain prior record-list output.\n'
            assert b'TYPE MISMATCH' in data(directory+'/compiler.stderr')
        else:
            assert r['exit']==r['native_exit']==0 and rows[directory+'/program']['sha256']==r['output_sha256']
            head=data(directory+'/program')[:64];assert head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==183
    assert rejected==1
assert neighbors['terminal'] and neighbors['image_before']==neighbors['image_after']==final['profiles'][0]['image_sha256']
assert neighbors['tools_before']==neighbors['tools_after'] and all(c['exit']==0 for c in neighbors['commands'])
for name,h in neighbors['tools_before'].items():assert rows['tool/'+name.lstrip('/')]['sha256']==h
assert b'Ran 21 tests' in data('journal/v2-enum.stderr') and data('journal/v2-enum.stderr').endswith(b'OK\n')
assert b'Ran 6 tests' in data('journal/v2-clang.stderr') and data('journal/v2-clang.stderr').endswith(b'OK\n')
clang=[n for n in rows if n.startswith('journal/v2-clang-cases/') and n.endswith('/result.json')];assert len(clang)==6
for name in clang:
    r=js(name);assert r['compiler_sha256']==final['profiles'][0]['image_sha256']
    if not r['reject']:assert r['native_exit']==r['exit']==0 and b'Ubuntu clang version 18.1.3' in data(name.rsplit('/',1)[0]+'/program')
assert b'Ubuntu clang version 18.1.3' in data('journal/clang-image-comment.stdout')
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
for name,record in [('src/typechecker.c','journal/typechecker-v2.c'),('tests/test_cseed_record_lists.py','journal/source/tests/test_cseed_record_lists.py')]:
    assert subprocess.check_output(['git','show',m['source_pin']+':'+name],cwd=repo)==data(record)
changes=subprocess.check_output(['git','diff','--name-only',m['baseline_pin'],m['source_pin'],'--','src','tests','Makefile.gnu','docs/ROADMAP.md'],cwd=repo,text=True).splitlines();assert changes==['Makefile.gnu','docs/ROADMAP.md','src/typechecker.c','tests/test_cseed_record_lists.py']
repair=m['packaging_reference_repair'];assert not repair['native_bytes_changed'] and repair['corrected_reused_references']==4553 and repair['initial_sha256']==sha(data('packaging/initial-manifest.json'))
print('PASS integrity: %d records; original C refusal and first sanitizer leak retained. Final six controls pass normal/SAN/O3 and Clang18; original21 enum neighbors pass. Broader source/platform/release acceptance stays open.'%len(rows))
