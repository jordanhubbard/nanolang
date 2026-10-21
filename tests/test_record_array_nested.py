"""I qualify direct nested proof, public transport, real VM and native lifetimes."""
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from tests.native_process import run_captured
from tests.test_record_array_vm import PROVIDERS

ROOT = Path(__file__).resolve().parents[1]


class RecordArrayNested(unittest.TestCase):
    def test_public_nested_consumers(self):
        self.qualify(False)

    def test_private_nested_consumers(self):
        self.qualify(True)

    def qualify(self, private, leaf=None):
        artifacts = Path(tempfile.mkdtemp(prefix='nano-record-array-nested-'))
        print(f'I retain nested qualification at {artifacts}', flush=True)
        def run(name, command, build=False):
            result = run_captured(command, cwd=ROOT, timeout=int(os.environ.get('CMD_TIMEOUT', '1800')) if build else 180)
            (artifacts / (name + '.stdout')).write_bytes(result.stdout)
            (artifacts / (name + '.stderr')).write_bytes(result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout.decode(errors='replace') + result.stderr.decode(errors='replace'))
            return result
        cc = shlex.split(os.environ.get('RECORD_ARRAY_NESTED_CC', 'cc'))
        flags = [*shlex.split(os.environ.get('RECORD_ARRAY_NESTED_CFLAGS', '')), '-std=c11',
                 '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                 '-fno-sanitize-recover=all', '-Isrc', '-Isrc/nanoisa']
        if leaf is not None:
            flags += ['-DNESTED_TEST_LEAF='+leaf]
        if private:
            flags += ['-DNANO_RECORD_ARRAY_PRIVATE_RUNTIME']
        providers = list(dict.fromkeys([*PROVIDERS, 'src/nanoisa/nvm_v2_convert.c',
            'src/nanoisa/nvm_v2_module.c', 'src/nanoisa/nvm_format.c', 'src/nanoisa/nvm2c.c']))
        names = {Path(source).stem for source in providers}
        common = [obj for obj in shlex.split(os.environ['RECORD_ARRAY_NESTED_OBJECTS']) if Path(obj).stem not in names]
        common = [p for p in common if Path(p).suffix != '.a'] + [p for p in common if Path(p).suffix == '.a']
        ldflags = shlex.split(os.environ.get('RECORD_ARRAY_NESTED_LDFLAGS', '-lm'))
        objects = []
        for source in providers:
            if source == 'src/nanovm/vm.c':
                continue  # The harness includes it for scoped overflow controls.
            obj = artifacts / (Path(source).stem + '.o')
            hooks = ['-DRA_ALLOC_WRAP', '-include', 'tests/nanoisa/record_array_alloc.h'] if source in PROVIDERS else []
            run(obj.stem + '-build', [*cc, *flags, *hooks, '-c', source, '-o', str(obj)], True)
            objects.append(str(obj))
        exe = artifacts / 'nested-public'
        run('harness-build', [*cc, *flags, '-DRA_NESTED_WHITEBOX', 'tests/nanoisa/test_record_array_nested.c',
            'tests/nanoisa/record_array_alloc.c', *objects, *common, *ldflags, '-o', str(exe)], True)
        run('public-nested', [str(exe), str(artifacts)])
        run('allocation-prefixes', [str(exe), 'allocations'])
        sources = sorted(artifacts.glob('nested-*.c'))
        self.assertEqual(len(sources), 13 if leaf=='TAG_FLOAT' else 6)
        for source in sources:
            for opt in ('-O0', '-O2'):
                name = source.stem + opt[1:];native = artifacts / name
                run(name + '-build', [*cc, '-std=c11', opt, '-Wall', '-Wextra', '-Werror',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all', str(source),
                    str(ROOT / 'bin/nano_aot_runtime.o'), *ldflags, '-o', str(native)], True)
                run(name + '-execute', [str(native)])


if __name__ == '__main__':
    unittest.main()
