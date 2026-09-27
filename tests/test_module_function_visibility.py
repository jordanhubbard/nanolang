"""I qualify the actual C module helpers with ordinary complete provider objects."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import native_sdk_runner
ROOT=Path(__file__).resolve().parents[1]

class ModuleFunctionVisibility(unittest.TestCase):
    def test_owner_preflight_and_alias_allocation(self):
        work=Path(tempfile.mkdtemp(prefix='nano-module-visibility-',dir=os.environ.get('NANO_SPLIT_REPORT_DIR')))
        print('I retain module visibility controls at',work,flush=True)
        providers=[Path(p).resolve() for p in shlex.split(os.environ['NANO_VISIBILITY_OBJECTS'])]
        self.assertTrue(providers)
        self.assertFalse(any(p.name in ('main.o','module.o') for p in providers))
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
        (work/'ordinary-providers-before.json').write_text(json.dumps(before,indent=2)+'\n')
        try:
            cc=shlex.split(os.environ['NANO_SPLIT_CC'])
            flags=shlex.split(os.environ['NANO_SPLIT_CFLAGS'])
            links=shlex.split(os.environ['NANO_SPLIT_LDFLAGS'])
            exe=work/'module-visibility'
            native_sdk_runner.run(work,'build',[*cc,*flags,ROOT/'tests/module_function_visibility.c',*providers,*links,'-o',exe],ROOT,timeout=300)
            out,err,_=native_sdk_runner.run(work,'run',[exe],ROOT,timeout=60)
            self.assertEqual(out,b'I checked exact function owners, selective visibility and both alias allocation failures.\n')
            self.assertIn(b'I cannot import private function',err)
            (work/'scope.json').write_text(json.dumps({'scope':'actual included module TU, ordinary complete hashed providers',
                'failure':'both strdup sites in alias-copy helper, unchanged byte sentinel and zero live allocations after each failure, normal recovery',
                'exclusions':'not full provider sanitization, general environment publication rollback, or type import visibility'},indent=2)+'\n')
        finally:
            after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
            (work/'ordinary-providers-after.json').write_text(json.dumps(after,indent=2)+'\n')
            self.assertEqual(after,before)
