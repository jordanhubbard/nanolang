"""I prepare a real generated profile input without changing shared build outputs."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import unittest

from scripts import fast_gate as gate


class ProfilePreparation(unittest.TestCase):
    def test_managed_header_uses_the_real_private_producer(self):
        for variable, default in (('NMS_RUNTIME_CLANG', 'clang'), ('NMS_RUNTIME_OPT', 'opt')):
            executable = shlex.split(os.environ.get(variable, default))[0]
            if not shutil.which(executable):
                self.skipTest('I require the actual clang/opt runtime producer for this control.')
        compiler = shutil.which('cc')
        if not compiler:
            self.skipTest('I require a C compiler for the generated-header consumer.')
        shared = gate.ROOT / 'obj/nanoisa/managed_runtime_ir.h'
        before = hashlib.sha256(shared.read_bytes()).hexdigest() if shared.exists() else None
        with tempfile.TemporaryDirectory(prefix='nano-real-profile-input-') as directory:
            empty = Path(directory) / 'empty'
            empty.mkdir()
            command = [compiler, '-std=c99', '-D_GNU_SOURCE', '-O1', '-Wall', '-Wextra', '-Werror',
                       '-Isrc', '-Isrc/nanoisa', '-I', str(empty), '-S',
                       'src/nanoisa/nvm2llvm.c', '-o', '/dev/null']
            refused = gate.run(command)
            self.assertNotEqual(refused[1], 0)
            self.assertIn('managed_runtime_ir.h', refused[2])
            prepared, evidence = gate.prepare_profile_inputs([command], directory)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]['returncode'], 0)
            header = Path(evidence[0]['header'])
            self.assertTrue(header.is_relative_to(directory))
            self.assertEqual(hashlib.sha256(header.read_bytes()).hexdigest(), evidence[0]['header_sha256'])
            manifest = json.loads(header.with_suffix('.json').read_text())
            self.assertEqual(set(manifest['variants']), {'native', 'wasm32'})
            for source, identity in manifest['sources'].items():
                self.assertEqual(hashlib.sha256((gate.ROOT / source).read_bytes()).hexdigest(), identity)
            accepted = gate.run(prepared[0])
            self.assertEqual(accepted[1], 0, accepted[2])
        after = hashlib.sha256(shared.read_bytes()).hexdigest() if shared.exists() else None
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
