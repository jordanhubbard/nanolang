#!/usr/bin/env python3
"""I check fixture skip linkage without claiming a native Darwin run."""
from pathlib import Path
import gzip,hashlib,json,re,subprocess
base=Path(__file__).resolve().parent;m=json.loads((base/'manifest.json').read_text());h=lambda b:hashlib.sha256(b).hexdigest();p=base/m['base_manifest']['path'];assert h(p.read_bytes())==m['base_manifest']['sha256'];prior=json.loads(p.read_text());rows={}
for r in m['files']:
 z=(base/r['artifact']).read_bytes();assert h(z)==r['compressed_sha256'];b=gzip.decompress(z);assert h(b)==r['sha256'] and len(b)==r['bytes'];rows[r['path']]=b
for r in m['git_sources']:assert subprocess.check_output(['git','show',m['source_pin']+':'+r['path']],cwd=base)==rows['source/'+r['path']] and h(rows['source/'+r['path']])==r['sha256']
assert not subprocess.check_output(['git','diff','--name-only',prior['source_pin'],m['source_pin'],'--',':(top)src',':(top)src_nano',':(top)modules',':(top)Makefile.gnu'],cwd=base)
assert subprocess.check_output(['git','diff','--name-only',prior['source_pin'],m['source_pin'],'--',':(top)tests'],cwd=base).decode().splitlines()==['tests/test_eval_carrier_native_bytes.c']
oldrow=next(r for r in prior['files'] if r['path']=='source/tests/test_eval_carrier_native_bytes.c');old=gzip.decompress((p.parent/oldrow['artifact']).read_bytes());assert h(old)==oldrow['sha256'];new=rows['source/tests/test_eval_carrier_native_bytes.c']
a=b'''static size_t allocated(void) { return 0; }
#endif
''';b=b'''int g_argc;
char **g_argv;
int main(void) { puts("I skip native allocation accounting without glibc or ASan statistics.");return 0; }
#endif
#if defined(__GLIBC__) || defined(__SANITIZE_ADDRESS__) || defined(NANO_SCRATCH_ASAN)
''';restored=new.replace(a,b)
header=b'''int main(void) {
#if !defined(__GLIBC__) && !defined(__SANITIZE_ADDRESS__) && !defined(NANO_SCRATCH_ASAN)
    puts("I skip native allocation accounting without glibc or ASan statistics.");return 0;
#else
''';restored=restored.replace(header,b'int main(void) {\n');assert restored.endswith(b'    return result;\n#endif\n}\n');restored=restored.removesuffix(b'    return result;\n#endif\n}\n')+b'    return result;\n}\n\n#endif\n';assert restored==old
assert rows['journal/old-no-counter.c']==old.replace(b'#elif defined(__GLIBC__)',b'#elif 0').replace(b'#if defined(__GLIBC__) || defined(__SANITIZE_ADDRESS__) || defined(NANO_SCRATCH_ASAN)',b'#if 0')
assert rows['journal/new-no-counter.c']==new.replace(b'#elif defined(__GLIBC__)',b'#elif 0').replace(b'#if !defined(__GLIBC__) && !defined(__SANITIZE_ADDRESS__) && !defined(NANO_SCRATCH_ASAN)',b'#if 1')
q=json.loads(rows['journal/qualified.json']);assert q['source_sha256']==h(new);assert len(q['commands'])==6 and all(r['exit']==0 for r in q['commands']);assert json.loads(rows['journal/old-link.json'])['exit']==1 and b'undefined reference' in rows['journal/old-link.stderr'] and b'nano_test_' in rows['journal/old-link.stderr']
assert rows['journal/no-counter-run.stdout']==b'I skip native allocation accounting without glibc or ASan statistics.\n' and not rows['journal/no-counter-run.stderr']
for mode in ['normal','san']:
 text=rows['journal/'+mode+'-run.stdout'];q=re.search(rb'positive=(\d+).*delta=(-?\d+) calls=(\d+)',text);assert q and int(q[1])>=2097152 and int(q[2])==0 and int(q[3])==128
 for name in [mode+'-counter']:
  b=rows['journal/'+name];assert b.startswith(b'\x7fELF') and int.from_bytes(b[18:20],'little')==183
assert m['acceptance']==dict(forced_no_counter_link_and_skip=True,original_normal_sanitizer_heap_controls=True,darwin_execution=False,product_source_change=False,default_activation=False,whole_release=False)
print(f'PASS {len(rows)} records; old link refusal retained; forced skip and unchanged normal/sanitizer native-byte criteria pass; no product source change or Darwin execution claim')
