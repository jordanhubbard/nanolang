"""I retain actual extern declaration ownership without invoking an external ABI."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests import native_sdk_runner
ROOT=Path(__file__).resolve().parents[1]

class ExternDeclarationOwners(unittest.TestCase):
    def test_exact_owners_signatures_and_label_failures(self):
        work=Path(tempfile.mkdtemp(prefix='nano-extern-owners-',dir=os.environ.get('NANO_SPLIT_REPORT_DIR')))
        print('I retain extern declaration controls at',work,flush=True)
        providers=[Path(p).resolve() for p in shlex.split(os.environ['NANO_SPLIT_CONTEXT_OBJECTS'])]
        self.assertTrue(providers)
        self.assertFalse(any(p.name in ('main.o','typechecker.o') for p in providers))
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
        (work/'ordinary-providers-before.json').write_text(json.dumps(before,indent=2)+'\n')
        try:
            cc=shlex.split(os.environ['NANO_SPLIT_CC'])
            flags=shlex.split(os.environ['NANO_SPLIT_CFLAGS'])
            links=shlex.split(os.environ['NANO_SPLIT_LDFLAGS'])
            exe=work/'extern-owners'
            native_sdk_runner.run(work,'build',[*cc,*flags,ROOT/'tests/extern_declaration_owners.c',*providers,*links,'-o',exe],ROOT,timeout=300)
            out,err,_=native_sdk_runner.run(work,'run',[exe],ROOT,timeout=60)
            self.assertEqual(out,b'I retained exact extern declaration owners, complete signatures and unpublished failed labels.\n')
            self.assertEqual(err,b'')
            (work/'scope.json').write_text(json.dumps({'scope':'actual included C checker; ordinary complete providers; no foreign invocation', 'controls':'both module registration orders, index refresh, same-owner idempotence, complete array/callback/tuple mismatches, both staged label allocation failures and recovery', 'exclusion':'not a full allocator sweep or historical collector OOM policy change'},indent=2)+'\n')
        finally:
            after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in providers}
            (work/'ordinary-providers-after.json').write_text(json.dumps(after,indent=2)+'\n')
            self.assertEqual(after,before)
