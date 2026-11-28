"""I verify unused-generator red/green and keep record-list refusals visible."""
from pathlib import Path
import gzip,hashlib,json,subprocess,shlex
base=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha((base/'manifest.json').read_bytes())=='96d5f9b7d4b2ea3fe880717c7f52f4cb8dcd697164a1934902d72312f9b6150b'
m=json.loads((base/'manifest.json').read_bytes());rows={r['path']:r for r in m['files']};assert len(rows)==len(m['files'])
for h,c in m['chunks'].items():
 z=(base/'objects'/(h+'.gz')).read_bytes();assert len(z)==c['compressed_bytes'] and sha(z)==c['compressed_sha256'];raw=gzip.decompress(z);assert len(raw)==c['bytes'] and sha(raw)==h
assert {p.stem for p in (base/'objects').glob('*.gz')}==set(m['chunks'])
def blocks(r):
 for h in r['chunks']:yield gzip.decompress((base/'objects'/(h+'.gz')).read_bytes())
def data(n):return b''.join(blocks(rows[n]))
def js(n):return json.loads(data(n))
for r in rows.values():
 digest=hashlib.sha256();size=0
 for raw in blocks(r):digest.update(raw);size+=len(raw)
 assert size==r['bytes'] and digest.hexdigest()==r['sha256'],r['path']
assert not m['qualification'] and not m['whole_release']
red=js('journal/literal-schema-ready/result.json');green=js('journal/candidate-result.json');providers=js('providers/result.json')
assert red['terminal'] and green['terminal'] and red['source_pin']==providers['source_pin']
assert red['providers']==red['providers_after']
assert [c['exit'] for c in red['commands']]==[0,-6]
assert b'access(marker, F_OK)' in data('journal/literal-schema-ready/run.stderr')
redsource=data('journal/literal-schema-ready/fixture.c');candidate=data('candidate/tests/test_module_list_preparation.c')
assert sha(redsource)==red['test_sha256'] and sha(candidate)==green['test_sha256']
assert candidate.replace(b'/* I compile literals and schema lists without unused global generators. */',b'/* I compile inline lists without publishing unused global temporary files. */').replace(b'I preserve prior output and schema list declarations without launching a generator.',b'I preserve prior output and use inline list definitions without launching the generator.')==redsource
marker=data('journal/literal-schema-ready/nano-list-preparation-PEwSvr/generator-called')
assert b'UnusedMarker' in marker and b'LexerToken' in marker
for name in ['inline.o','schema.o']:assert rows['journal/literal-schema-ready/nano-list-preparation-PEwSvr/'+name]['bytes']>0
assert data('journal/literal-schema-ready/nano-list-preparation-PEwSvr/rejected.o')==b'retained object bytes\n'
assert sha(data('candidate/src/module.c'))==green['source_sha256']
old=data('baseline/src/module.c');new=data('candidate/src/module.c');removed=data('journal/removed-block.c')
assert old.count(removed)==1
replacement=b"    /* I use this unit's schema runtime headers and inline list definitions.\n     * Unused global /tmp files do not supply either kind of declaration. */\n\n"
assert old.replace(removed,replacement)==new
assert [p['mode'] for p in green['profiles']]==['normal','san','production']
work=Path(green['work'])
for p,original in zip(green['profiles'],providers['profiles']):
 mode=p['mode'];assert p['mode']==original['mode'] and p['root']==original['tree']
 assert p['providers']==p['providers_after']
 for name,h in p['providers'].items():assert h==original['outputs'][name]['sha256']
 assert [c['exit'] for c in p['commands']]==[0,0,0]
 assert rows['journal/'+work.name+'/'+mode+'-test']['sha256']==p['binary_sha256']
 assert b'without launching a generator' in data('journal/'+work.name+'/'+mode+'-run.stdout')
 assert not any('generator-called' in n for n in rows if n.startswith('journal/'+work.name+'/'))
 originalargs=shlex.split(next(l for l in data('providers/'+mode+'-focused.stdout').decode().splitlines() if '-o obj/test_module_rejection_rollback ' in l));args=p['commands'][1]['args'];i=originalargs.index('-o')
 assert originalargs[:i]==args[:i] and originalargs[i+3:]==['obj/module.o' if a==str(work/(mode+'-module.o')) else a for a in args[i+3:]]
neighbors=js('journal/neighbor-result.json');assert len(neighbors)==6 and all(c['exit']==0 for c in neighbors)
cleanup=[n for n in rows if n.startswith('journal/cleanup-') and n.endswith('/result.json')];assert len(cleanup)==1
assert js(cleanup[0])['exit']==0 and js(cleanup[0])['fixture_paths_remaining']==[]
for label in ['corrected','public-control']:
 r=js('journal/'+label+'/result.json');assert r['commands'][0]['exit']==0 and r['commands'][-1]['exit']!=0
 err=data('journal/'+label+'/run.stderr');assert b'ModuleListProbe' in err
 assert any(n.startswith('journal/'+label+'/') and n.endswith('/source.c') for n in rows)
repo=subprocess.check_output(['git','-C',str(base),'rev-parse','--show-toplevel'],text=True).strip()
assert subprocess.check_output(['git','show',m['source_pin']+':src/module.c'],cwd=repo)==old
for name in ['src/module.c','Makefile.gnu','tests/test_module_list_preparation.c']:
 assert subprocess.check_output(['git','show','5bf50072752f6e88bad274058663e6e7effc1f89:'+name],cwd=repo)==data('candidate/'+name),name
print('PASS integrity: %d records; old generator side effect reproduced; normal/SAN/O3 list and original rollback cases pass; cleanup passes. Private/public record-list C refusals retained under#1248; full platform/source qualification remains open.'%len(rows))
