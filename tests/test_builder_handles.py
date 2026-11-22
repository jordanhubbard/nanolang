"""I qualify the actual Builder handle implementation without source-family admission."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BuilderHandles(unittest.TestCase):
    def test_actual_native_boundary(self):
        work = Path(tempfile.mkdtemp(prefix='builder-handles-', dir=ROOT/'.test_output'))
        print('I retain native Builder evidence at', work, flush=True)
        cc = shlex.split(os.environ.get('CC', 'cc'))
        compiler = Path(shutil.which(cc[0]) or cc[0]).resolve(strict=True)
        inputs = ['tests/test_builder_handles.py', 'tests/test_builder_handles.c',
                  'modules/std/collections/collections.c', 'modules/std/collections/collections.h',
                  'modules/std/collections/module.json', 'src/runtime/dyn_array.c',
                  'src/runtime/gc.c', 'src/runtime/gc_struct.c']
        before = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs}
        compiler_hash = hashlib.sha256(compiler.read_bytes()).hexdigest()
        version = subprocess.check_output([*cc, '--version'], text=True, timeout=10)
        operations, products = [], {}
        (work/'inputs-before.json').write_text(json.dumps(before, indent=2)+'\n')

        def preserve_custody():
            after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs}
            images = {name: {'sha256_before': row['sha256_before'],
                            'sha256_after': hashlib.sha256((work/name).read_bytes()).hexdigest()}
                      for name, row in products.items()}
            (work/'custody-after.json').write_text(json.dumps({'inputs_before': before,
                'inputs_after': after, 'products': images, 'compiler_before': compiler_hash,
                'compiler_after': hashlib.sha256(compiler.read_bytes()).hexdigest()}, indent=2)+'\n')
            self.assertEqual(before, after)
            self.assertEqual(compiler_hash, hashlib.sha256(compiler.read_bytes()).hexdigest())
            for row in images.values(): self.assertEqual(row['sha256_before'], row['sha256_after'])
        self.addCleanup(preserve_custody)

        def command(label, argv, expected=0, fault=False):
            environment = dict(os.environ)
            if fault:
                # My existing growth refusal exits the process with live state.
                # I check its diagnostic and status separately from clean teardown.
                options = environment.get('ASAN_OPTIONS', '')
                environment['ASAN_OPTIONS'] = options+':detect_leaks=0'
            result = subprocess.run(list(map(str, argv)), cwd=ROOT, env=environment,
                                    capture_output=True, timeout=60)
            (work/(label+'.stdout')).write_bytes(result.stdout)
            (work/(label+'.stderr')).write_bytes(result.stderr)
            operations.append({'label': label, 'argv': list(map(str, argv)),
                               'returncode': result.returncode, 'fault_child': fault})
            (work/'operations.json').write_text(json.dumps(operations, indent=2)+'\n')
            self.assertEqual(result.returncode, expected, result.stdout+result.stderr)
            self.assertNotRegex(result.stderr.decode(errors='replace'),
                                r'ERROR: AddressSanitizer|SUMMARY: AddressSanitizer|ThreadSanitizer:|runtime error:')
            return result

        for name, flags in [('plain', []),
                            ('sanitized', ['-fsanitize=address,undefined', '-fno-sanitize-recover=all']),
                            ('threaded', ['-fsanitize=thread'])]:
            binary = work/name
            command('build-'+name, [*cc, '-std=c99', '-O1', '-g', '-Wall', '-Wextra',
                    '-Werror', '-UNDEBUG', '-pthread', '-Isrc', *flags,
                    'tests/test_builder_handles.c', 'src/runtime/dyn_array.c',
                    'src/runtime/gc.c', 'src/runtime/gc_struct.c', '-lm', '-o', binary])
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            products[name] = {'sha256_before': digest}
            (work/'products-before.json').write_text(json.dumps(products, indent=2)+'\n')
            self.assertIn(binary.read_bytes()[:4], (b'\x7fELF', b'\xcf\xfa\xed\xfe'))
            observed = command('run-'+name, [binary])
            self.assertIn(b'I pass checked Builder identity', observed.stdout)
            if name != 'threaded':
                for fault, diagnostic in [('growth-oom', b'I could not allocate StringBuilder growth.'),
                                          ('overflow', b'I cannot grow StringBuilder beyond its length bounds.')]:
                    observed = command(name+'-'+fault, [binary, fault], expected=1, fault=True)
                    self.assertIn(diagnostic, observed.stderr)
                    if fault == 'growth-oom':
                        self.assertIn(b'I retain prior Builder contents on growth refusal.', observed.stderr)
            self.assertEqual(hashlib.sha256(binary.read_bytes()).hexdigest(), digest)
            products[name] = {'sha256_before': digest, 'sha256_before_after': digest}
        self.assertEqual(before, {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs})
        self.assertEqual(compiler_hash, hashlib.sha256(compiler.read_bytes()).hexdigest())
        (work/'inputs-before-after.json').write_text(json.dumps(before, indent=2)+'\n')
        (work/'products.json').write_text(json.dumps(products, indent=2)+'\n')
        (work/'compiler.json').write_text(json.dumps({'argv': cc, 'launcher': str(compiler),
            'sha256_before_after': compiler_hash, 'version': version,
            'scope': 'I pin selected source and compiler images, not the full system header/toolchain closure.'}, indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
