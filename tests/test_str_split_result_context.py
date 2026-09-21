"""I isolate the C field adapter from the unchanged public scalar-borrow restriction."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import native_sdk_runner
ROOT=Path(__file__).resolve().parents[1]

class SplitResultContext(unittest.TestCase):
    def test_actual_field_assignment_and_binding_refusal(self):
        work=Path(tempfile.mkdtemp(prefix='nano-split-context-',dir=os.environ.get('NANO_SPLIT_REPORT_DIR')))
        print('I retain direct split context controls at',work,flush=True)
        providers=[Path(p).resolve() for p in shlex.split(os.environ['NANO_SPLIT_CONTEXT_OBJECTS'])]
        self.assertTrue(providers)
        self.assertFalse(any(p.name in ('main.o','typechecker.o') for p in providers))
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
        (work/'ordinary-providers-before.json').write_text(json.dumps(before,indent=2)+'\n')
        try:
            cc=shlex.split(os.environ['NANO_SPLIT_CC'])
            flags=shlex.split(os.environ['NANO_SPLIT_CFLAGS'])
            links=shlex.split(os.environ['NANO_SPLIT_LDFLAGS'])
            exe=work/'result-context'
            native_sdk_runner.run(work,'build',[*cc,*flags,ROOT/'tests/str_split_result_context.c',*providers,*links,'-o',exe],ROOT,timeout=300)
            out,err,_=native_sdk_runner.run(work,'run',[exe],ROOT,timeout=60)
            self.assertEqual(out,b'I checked the string-array field adapter and binding-refusal publication boundary.\n')
            self.assertIn(b'I require array<string> for this string-array result.',err)
            (work/'scope.json').write_text(json.dumps({'scope':'actual included C typechecker adapter; ordinary hashed complete providers; public array-field borrowing remains refused',
                'failure':'injected false return at metadata binding boundary, not a malloc-site sweep','cases':['STRING acceptance and repeated exact metadata','INT refusal','binding refusal leaves no row and flags preparation failure']},indent=2)+'\n')
        finally:
            after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
            (work/'ordinary-providers-after.json').write_text(json.dumps(after,indent=2)+'\n')
            self.assertEqual(after,before)
