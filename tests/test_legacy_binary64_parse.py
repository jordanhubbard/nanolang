"""I preserve consumed-prefix boundaries and actual legacy source execution."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class LegacyBinary64Parse(unittest.TestCase):
    def command(self, args):
        result = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_checked_endpoint_and_prefix_values(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / 'endpoints.c'
            source.write_text(r'''
#include "runtime/binary64_parse.h"
#include <assert.h>
static void check(const char *text, uint32_t length, uint32_t endpoint, uint64_t expected) {
 uint64_t bits=99; uint32_t end=99;
 assert(nbp_parse_end((const unsigned char *)text,length,&bits,&end));
 assert(bits==expected && end==endpoint);
 uint64_t prefix; assert(nbp_parse((const unsigned char *)text,length,&prefix));
 assert(prefix==bits);
}
#define C(s,e,b) check(s,sizeof(s)-1,e,UINT64_C(b))
int run(void) {
 C("",0,0); C("  +",0,0); C("-.",0,0); C("x",0,0);
 C("1e+",1,0x3ff0000000000000); C("0x1p-",3,0x3ff0000000000000);
 C("1e+2tail",4,0x4059000000000000); C("0x1.8p+1!",8,0x4008000000000000);
 C("  -0e+99",8,0x8000000000000000); C("0x0p-99",7,0);
 C("0e+",1,0); C("0x",1,0); C("0x.",1,0);
 C("2.5\0tail",3,0x4004000000000000); C("2.5 ",3,0x4004000000000000);
 C("Infinity!",8,0x7ff0000000000000); C("infinit",3,0x7ff0000000000000);
 C(" -INF",5,0xfff0000000000000); C("nan",3,0x7ff8000000000000);
 C("nan()!",5,0x7ff8000000000000); C("nan(_abc)",9,0x7ff8000000000000);
 C("nan(+1)",3,0x7ff8000000000000); C("nan(123",3,0x7ff8000000000000);
 C("nan(0x123)",10,0x7ff8000000000123);
 C("nan(184467440737095516160000)",29,0x7fffffffffffffff);
 C("+NaN(foo)tail",9,0x7ff8000000000000);
 return 0;
}
#ifndef __wasm32__
int main(void){return run();}
#endif
''')
            exe = work / 'endpoints'
            self.command(['clang', *shlex.split(os.environ.get('NMS_NATIVE_CLANG_FLAGS','')),
                          '-O2','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all',
                          '-I'+str(ROOT/'src'),source,'-o',exe])
            self.command([exe])

    def test_all_source_compilers_keep_prefix_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'program'
            source = ROOT / 'tests/unit/test_legacy_binary64_parse.nano'
            for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
                with self.subTest(compiler=compiler):
                    self.command([ROOT/'bin'/compiler,source,'-o',output])
                    self.command([output])

if __name__ == '__main__': unittest.main()
