"""I qualify only explicit private adapters, with durable first outcomes."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

NATIVE = r'''#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
static size_t oa_live,oa_attempts,oa_fail,oa_hits;
static int oa_printed;
static FILE *oa_output;
static void *oa_malloc(size_t n){if(++oa_attempts==oa_fail){oa_hits++;return NULL;}void *p=malloc(n);if(p)oa_live++;return p;}
static void *oa_calloc(size_t n,size_t s){if(++oa_attempts==oa_fail){oa_hits++;return NULL;}void *p=calloc(n,s);if(p)oa_live++;return p;}
static void oa_free(void *p){if(p){assert(oa_live);oa_live--;free(p);}}
static int oa_fprintf(FILE *out,const char *format,...){va_list a;va_start(a,format);int n=vfprintf(out,format,a);va_end(a);if(out==oa_output)oa_printed=1;return n;}
#define malloc oa_malloc
#define calloc oa_calloc
#define free oa_free
#define fprintf oa_fprintf
#undef stdout
#define stdout oa_output
#define NVM2C_NO_MAIN
#include "GENERATED"
#undef fprintf
#undef malloc
#undef calloc
#undef free
static void run(size_t fault){
 oa_output=tmpfile();assert(oa_output);oa_printed=0;oa_attempts=oa_hits=0;oa_fail=fault;
 int64_t value=-91;int status=nvm_owned_entry(&value);oa_fail=0;
 fprintf(stderr,"private native observed fault=%zu allocations=%zu hits=%zu status=%d value=%lld live=%zu\n",fault,oa_attempts,oa_hits,status,(long long)value,oa_live);
 assert(!oa_live);assert(!fflush(oa_output));long bytes=ftell(oa_output);rewind(oa_output);
 if(oa_printed){assert(bytes==2);assert(fgetc(oa_output)=='7' && fgetc(oa_output)=='\n');}else assert(bytes==0);
 assert(!fclose(oa_output));oa_output=NULL;
 fprintf(stderr,"private native fault=%zu allocations=%zu hits=%zu status=%d prefix=%ld live=%zu\n",fault,oa_attempts,oa_hits,status,bytes,oa_live);
 if(oa_hits){assert(oa_hits==1 && status==1 && value==-91);}else{assert(status==WANTED && value==VALUE && oa_printed);}
}
int main(void){
 nown_string *s=nown_string_new((const unsigned char *)"held",4);assert(s);s->refs=SIZE_MAX;
 assert(!nown_retain((nown_value){.tag=5,.string=s}) && s->refs==SIZE_MAX);s->refs=1;nown_release((nown_value){.tag=5,.string=s});assert(!oa_live);
 run(0);int done=0;
 for(size_t fault=1;fault<256;fault++){run(fault);if(!oa_hits){done=1;break;}run(0);}assert(done);return 0;
}
'''

class PrivateOwnedArrayRuntime(unittest.TestCase):
    def command(self, args):
        p = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True, text=True, timeout=180)
        stem = self.work / f'{self.serial:03d}'
        self.serial += 1
        stem.with_suffix('.json').write_text(json.dumps(list(map(str, args))))
        stem.with_suffix('.log').write_text(p.stdout + p.stderr)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def test_private_vm_native(self):
        self.serial = 0
        supplied = os.environ.get('PRIVATE_OWNER_ARRAY_ARTIFACTS')
        self.work = Path(supplied) if supplied else Path(tempfile.mkdtemp(prefix='nano-private-owner-array-'))
        if supplied:
            self.work.mkdir(exist_ok=False, parents=True)
        cc = shlex.split(os.environ.get('CC', 'cc'))
        flags = shlex.split(os.environ.get('PRIVATE_OWNER_ARRAY_CFLAGS', ''))
        common = [*cc, *flags, '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-D_GNU_SOURCE', '-Isrc', '-Isrc/nanoisa', '-DNANO_OWNED_ARRAY_PRIVATE_RUNTIME']
        # Normal canonical objects must not export private entrypoints.
        for obj, symbol in [('obj/nanovm/vm.o', 'vm_execute_owned_array_private'), ('obj/nanoisa/nvm2c.o', 'nvm2c_emit_owned_array_private')]:
            self.assertNotIn(symbol, self.command(['nm', '-g', obj]).stdout)
        providers = shlex.split(os.environ['PRIVATE_OWNER_ARRAY_OBJECTS'])
        libraries = shlex.split(os.environ['PRIVATE_OWNER_ARRAY_LDFLAGS'])
        switch_only = os.environ.get("PRIVATE_OWNER_ARRAY_SWITCH_ONLY") == "1"
        for threaded in ((False,) if switch_only else (False, True)):
            phase = self.work / ('threaded' if threaded else 'switch')
            phase.mkdir()
            objects = []
            for name, source in [('vm', 'src/nanovm/vm.c'), ('nvm2c', 'src/nanoisa/nvm2c.c'), ('heap', 'src/nanovm/heap.c')]:
                obj = phase / (name + '.o')
                extra = (['-DNANO_COMPUTED_GOTO'] if threaded else ['-DNANO_NO_COMPUTED_GOTO']) if name == 'vm' else []
                if name == 'vm':
                    macros = self.command([*common, *extra, '-dM', '-E', source]).stdout
                    if threaded:
                        self.assertIn('#define NANO_COMPUTED_GOTO 1', macros)
                    else:
                        self.assertIn('#define NANO_NO_COMPUTED_GOTO 1', macros)
                        self.assertNotIn('#define NANO_COMPUTED_GOTO ', macros)
                if name == 'heap':
                    extra += ['-Dmalloc=private_array_malloc', '-Dcalloc=private_array_calloc', '-Drealloc=private_array_realloc']
                self.command([*common, *extra, '-c', source, '-o', obj])
                objects.append(obj)
            binary = phase / 'fixture'
            self.command([*common, 'tests/nanoisa/test_private_owned_array_runtime.c', *objects, *providers, *libraries, '-o', binary])
            run = self.command([binary, phase])
            rows = [line.split() for line in run.stdout.splitlines() if line.startswith('case ')]
            self.assertEqual(len(rows), 7)
            if threaded:
                for _, index, _ in rows:
                    self.assertEqual((phase/f'case{index}.c').read_bytes(), (self.work/'switch'/f'case{index}.c').read_bytes())
                continue
            if switch_only:
                continue
            for _, index, status in rows:
                harness = phase / f'native{index}.c'
                harness.write_text(NATIVE.replace('GENERATED', f'case{index}.c').replace('WANTED', status).replace('VALUE', '0' if status == '0' else '-91'))
                for opt in ('-O0', '-O2'):
                    binary = phase / f'native{index}{opt}'
                    self.command([*cc, *flags, '-std=c11', '-Wall', '-Wextra', '-Werror', opt, harness, '-o', binary])
                    self.command([binary])
