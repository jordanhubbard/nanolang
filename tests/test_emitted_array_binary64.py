#!/usr/bin/env python3
"""I qualify my real emitted array helpers without building a compiler stage."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('NANO_EMITTED_ARRAY_REPORT', tempfile.mkdtemp(prefix='nano-emitted-array-')))
OUT.mkdir(parents=True, exist_ok=True)
ORDER = os.environ.get('NANO_PROVIDER_ORDER', 'string-first')
assert ORDER in ['string-first','math-first']
CC = shlex.split(os.environ.get('CC', 'cc'))
records = []

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

files = ['src/stdlib_runtime.c', 'src/stdlib_runtime.h', 'tests/test_emitted_array_binary64.py',
         'tests/test_emitted_array_binary64.c', 'tests/test_aggregate_binary64_eval.c']
files += [str(p.relative_to(ROOT)) for p in (ROOT/'src/runtime').glob('*.[ch]')]
files += ['src/utf8.c', 'src/utf8.h', 'src/binary64_arithmetic_source.h',
          'src/binary64_arithmetic.h', 'src/binary64_bits.h', 'src/binary64_format.h']
identities = {str(ROOT/f): digest(ROOT/f) for f in files}
identities[str(Path(shutil.which(CC[0])).resolve())] = digest(shutil.which(CC[0]))

def run(label, args):
    start = time.monotonic()
    result = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, 'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1',
                                 'UBSAN_OPTIONS':'halt_on_error=1'}, timeout=180)
    (OUT/(label+'.log')).write_bytes(result.stdout)
    records.append(dict(label=label, command=list(map(str,args)), status=result.returncode,
                        seconds=round(time.monotonic()-start,3)))
    if result.returncode:
        raise RuntimeError(f'{label}: exit {result.returncode}; see {OUT}')
    return result.stdout.decode()

try:
    generator = OUT/'generator.c'
    generator.write_text('''#include <stdio.h>
#include "stdlib_runtime.h"
void sb_append(StringBuilder *sb, const char *s) { (void)sb; fputs(s,stdout); }
int main(int argc, char **argv) { (void)argv; StringBuilder sb={0};
if(argc>1) generate_math_utility_builtins(&sb);
generate_string_operations(&sb);
if(argc==1) generate_math_utility_builtins(&sb);
return ferror(stdout); }
''')
    run('generator-build', CC+['-std=c11','-D_DEFAULT_SOURCE','-O2','-Wall','-Wextra','-Werror',
        '-Isrc',str(generator),'src/stdlib_runtime.c','-o',str(OUT/'generator')])
    emitted = run('provider', [str(OUT/'generator')]+(['math-first'] if ORDER=='math-first' else []))
    includes = '''#include <stdarg.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <ctype.h>
#include "runtime/gc.h"
#include "runtime/dyn_array.h"
'''
    names = sorted(set(re.findall(r'^static\s+(?:inline\s+)?[\w\s*]+?\s+(\w+)\s*\([^;{}]*\)\s*\{', emitted, re.M)))
    retention = '\nstatic void retain_emitted_helpers(void) {\n'+''.join(f'    (void)&{n};\n' for n in names)+'}\n'
    (OUT/'emitted_runtime.h').write_text(includes+emitted+retention)
    oracle = (ROOT/'tests/test_aggregate_binary64_eval.c').read_text()
    oracle = oracle[oracle.index('typedef struct { unsigned operation;'):oracle.index('static double from_bits')]
    (OUT/'array_cases.h').write_text(oracle)
    runtime = ['src/runtime/dyn_array.c','src/runtime/gc.c','src/runtime/gc_struct.c',
               'src/runtime/nl_string.c','src/utf8.c']
    for opt in ['-O0','-O2']:
        label=opt[1:]
        run(label+'-build',CC+['-std=c11','-D_DEFAULT_SOURCE',opt,'-Wall','-Wextra','-Werror',
            '-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer',
            '-Isrc','-I'+str(OUT),'tests/test_emitted_array_binary64.c']+runtime+['-lm','-o',str(OUT/label)])
        output=run(label+'-run',[str(OUT/label)])
        assert output.startswith('PASS: 272 fixed-bit results, 12 empty outputs'), output
finally:
    unchanged=all(digest(f)==h for f,h in identities.items())
    report=dict(provider_order=ORDER,source=str(ROOT),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                identities=identities,unchanged=unchanged,records=records)
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(OUT)
    assert unchanged, 'Source or tool changed during qualification'
