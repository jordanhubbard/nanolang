"""I exercise shared array-field authority through actual public consumers."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from tests.test_record_array_vm import PROVIDERS as VM_PROVIDERS

ROOT = Path(__file__).resolve().parents[1]


class RecordArrayPublic(unittest.TestCase):
    def test_public_vm_bridge_and_sanitized_native(self):
        artifacts = Path(tempfile.mkdtemp(prefix='nano-record-array-public-'))
        print(f'I retain public-route evidence at {artifacts}', flush=True)
        def run(name, command, build=False):
            result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                    timeout=int(os.environ.get('CMD_TIMEOUT', '1800')) if build else 120)
            (artifacts / (name + '.stdout')).write_bytes(result.stdout)
            (artifacts / (name + '.stderr')).write_bytes(result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout.decode(errors='replace') + result.stderr.decode(errors='replace'))
            return result
        # I instrument the actual public VM and proof/transport providers,
        # keeping the ordinary ABI (no private-runtime compile definition).
        cc = shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_CC', 'cc'))
        providers = list(dict.fromkeys([*VM_PROVIDERS, 'src/nanoisa/nvm_v2_convert.c',
            'src/nanoisa/nvm_v2_module.c', 'src/nanoisa/nvm_format.c', 'src/nanoisa/nvm2c.c']))
        excluded = {Path(source).stem for source in providers}
        common = [obj for obj in shlex.split(os.environ['RECORD_ARRAY_PUBLIC_OBJECTS'])
                  if Path(obj).stem not in excluded]
        # I preserve single-pass Unix archive resolution after object users.
        common = [p for p in common if Path(p).suffix != '.a'] + [p for p in common if Path(p).suffix == '.a']
        flags = [*shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_CFLAGS', '')),
                 '-std=c11', '-D_DEFAULT_SOURCE', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                 '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-Isrc', '-Isrc/nanoisa']
        objects = []
        for source in providers:
            obj = artifacts / (Path(source).stem + '.o')
            run(obj.stem + '-instrument', [*cc, *flags, '-c', source, '-o', str(obj)], build=True)
            objects.append(str(obj))
        instrumented = artifacts / 'public-api-instrumented'
        run('instrumented-link', [*cc, *flags, 'tests/nanoisa/test_record_array_public.c',
            'tests/nanoisa/record_array_alloc.c', *objects, *common,
            *shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_LDFLAGS', '-lm')),
            '-o', str(instrumented)], build=True)
        run('public-api-instrumented', [str(instrumented), str(artifacts)])
        receipt = run('public-api', [str(ROOT / 'bin/test_record_array_public'), str(artifacts)])
        self.assertIn(b'public record-array checks', receipt.stdout)
        sources = sorted([*artifacts.glob('array-*.c'), *artifacts.glob('flow-*.c')])
        self.assertEqual(len(sources), 10)
        cc = shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_CC', 'cc'))
        for source in sources:
            for optimization in ('-O0', '-O2'):
                name = source.stem + optimization[1:]
                executable = artifacts / name
                run(name + '-build', [*cc, '-std=c11', optimization, '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all', str(source),
                    str(ROOT / 'bin/nano_aot_runtime.o'), '-lm',
                    *shlex.split(os.environ.get('RECORD_ARRAY_PUBLIC_LDFLAGS', '')),
                    '-o', str(executable)], build=True)
                run(name + '-execute', [str(executable)])


if __name__ == '__main__':
    unittest.main()
