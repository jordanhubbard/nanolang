"""I preserve declaration priority and lexical bindings in checked bytecode."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADERS = [Path(p) / 'sqlite3.h' for p in
           ('/opt/homebrew/include', '/usr/local/include', '/usr/include')]


def private_header(work, definitions):
    header = work / 'constants.h'
    header.write_text(definitions)
    # I use a private header without installing it into a system directory.
    include = next(p for p in (Path('/opt/homebrew/include'), Path('/usr/local/include'),
                               Path('/usr/include')) if p.is_dir())
    (work / 'module.json').write_text(json.dumps({
        'name': 'header_values', 'headers': [os.path.relpath(header, include)]}))
    dependency = work / 'values.nano'
    dependency.write_text('pub fn declared()->int { return 17 }\n'
                          'shadow declared { assert (== (declared) 17) }\n')
    return dependency


@unittest.skipUnless(any(p.is_file() for p in HEADERS),
                     'I require SQLite headers in the module constant search paths.')
class HeaderConstantFunctions(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nano-header-functions-')
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.environment = {**os.environ, 'NANO_BUILD_CACHE': str(self.work / 'cache')}

    def command(self, args):
        return subprocess.run(list(map(str, args)), cwd=ROOT, env=self.environment,
                              capture_output=True, text=True, timeout=120)

    def checked(self, args, stdout=None):
        result = self.command(args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if stdout is not None:
            self.assertEqual(result.stdout, stdout)

    def execute(self, source, stdout):
        module = self.work / 'out.nvm'
        self.checked([ROOT / 'bin/nano_virt', source, '--emit-nvm', '-o', module])
        self.checked([ROOT / 'bin/nano_vm', '--verify-only', module])
        self.checked([ROOT / 'bin/nano_vm', module], stdout)

    def source(self, body):
        path = self.work / 'main.nano'
        path.write_text('unsafe module "modules/sqlite/sqlite.nano" as sqlite\n' + body)
        return path

    def test_original_sqlite_example_and_dependency_shadows(self):
        self.execute(ROOT / 'examples/advanced/sqlite_simple.nano',
                     'prepared CRUD: Alice -> Alicia -> deleted\n')

    def test_callable_local_still_shadows_declared_function(self):
        source = self.source('''fn replacement()->int { return 37 }
shadow replacement { assert (== (replacement) 37) }
fn main()->int {
 assert (== (sqlite.SQLITE_OK) 0)
 let SQLITE_OK: fn() -> int = replacement
 assert (== (SQLITE_OK) 37)
 return 0
}
shadow main { assert (== (main) 0) }
''')
        self.execute(source, '')

    def test_unrelated_sqlite_header_value(self):
        source = self.source('''fn main()->int {
 assert (== SQLITE_BUSY 5)
 return 0
}
shadow main { assert (== (main) 0) }
''')
        self.execute(source, '')

    def test_noncallable_local_refuses_without_replacing_prior_output(self):
        source = self.source('''fn main()->int {
 let SQLITE_OK: int = 7
 return (SQLITE_OK)
}
shadow main { assert true }
''')
        for compiler, flags in (('nano_virt', ['--emit-nvm']), ('nanoc_c', [])):
            with self.subTest(compiler=compiler):
                output = self.work / (compiler + '.out')
                output.write_bytes(b'prior output\n')
                result = self.command([ROOT / 'bin' / compiler, source, *flags, '-o', output])
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('I require a function value for a bound call.', result.stderr)
                self.assertIn(str(source), result.stderr)
                self.assertEqual(output.read_bytes(), b'prior output\n')


class HeaderConstantValues(unittest.TestCase):
    compiler = 'nano_virt'

    def paired(self, source, work):
        module, generated, native = (work / name for name in ('out.nvm', 'out.c', 'out'))
        cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or os.environ.get('CC') or 'cc')
        environment = {**os.environ, 'NANO_BUILD_CACHE': str(work / 'cache'),
                       'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'}
        commands = [
            [ROOT / 'bin' / self.compiler, source, '--emit-nvm', '-o', module],
            [ROOT / 'bin/nano_vm', '--verify-only', module],
            [ROOT / 'bin/nano_vm', module],
            [ROOT / 'bin/nvm2c', module, '-o', generated],
            [*cc, '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
             '-fno-sanitize-recover=all', generated, '-ldl', '-lm', '-o', native],
            [native],
        ]
        for command in commands:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=environment,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, repr(command) + '\n' + result.stdout + result.stderr)

    @unittest.skipUnless(any(p.is_file() for p in HEADERS), 'I require SQLite headers.')
    def test_real_sqlite_header_value(self):
        with tempfile.TemporaryDirectory(prefix='nano-sqlite-header-') as temporary:
            work = Path(temporary).resolve()
            source = work / 'sqlite.nano'
            source.write_text('unsafe module "modules/sqlite/sqlite.nano" as sqlite\n'
                              'fn main()->int { assert (== SQLITE_BUSY 5) return 0 }\n'
                              'shadow main { assert (== (main) 0) }\n')
            self.paired(source, work)

    def test_compiler_support_query_preserves_snapshot_and_literal_contract(self):
        with tempfile.TemporaryDirectory(prefix='nano-header-query-') as temporary:
            work = Path(temporary).resolve()
            dependency = private_header(work, '#define HEADER_VALUE 012\n#define INVALID_VALUE 1 + 2\n')
            source = work / 'query.nano'
            source.write_text('module "modules/compiler_support/compiler_support.nano" as support\n'
                              'fn main()->int {\n'
                              f' let snapshot:string = (support.module_header_constants "{dependency}")\n'
                              ' assert (== snapshot "HEADER_VALUE=10\\n")\n'
                              ' assert (== (support.module_header_constants "") "")\n'
                              f' assert (== (support.module_header_constants "{work}") "")\n'
                              ' assert (== snapshot "HEADER_VALUE=10\\n")\n return 0\n}\n'
                              'shadow main { assert (== (main) 0) }\n')
            self.paired(source, work)

    def test_header_values_keep_lexical_bindings_in_shadows_vm_and_native(self):
        with tempfile.TemporaryDirectory(prefix='nano-header-values-') as temporary:
            work = Path(temporary).resolve()
            dependency = private_header(work,
                '#define HEADER_VALUE 41\n#define HEADER_NEGATIVE -7\n'
                '#define HEADER_HEX 0x1234\n#define declared 999\n'
                '#define HEADER_OCTAL 012\n#define HEADER_SUFFIX 73ULL /* units */\n'
                '#define HEADER_REVERSE_SUFFIX 23lu\n#define HEADER_ZERO 0u\n'
                '#define HEADER_HEX_MAX 0x7fffffffffffffffL\n'
                '#define HEADER_COMMENT 19 // units\n'
                '#define HEADER_MAX 9223372036854775807LL\n'
                '#define HEADER_MIN -9223372036854775808LL\n')
            source = work / 'main.nano'
            source.write_text(f'module "{dependency}" as values\n' + '''
let saved_header:int = HEADER_VALUE
fn read_header()->int { return HEADER_VALUE }
shadow read_header { assert (== (read_header) 41) }
fn local_header()->int { let HEADER_VALUE:int = 83 return HEADER_VALUE }
shadow local_header { assert (== (local_header) 83) }
fn parameter_header(HEADER_VALUE:int)->int { return HEADER_VALUE }
shadow parameter_header { assert (== (parameter_header 29) 29) }
fn captured_header()->int {
 let HEADER_VALUE:int = 61
 let read:fn() -> int = fn()->int { return HEADER_VALUE }
 return (read)
}
shadow captured_header { assert (== (captured_header) 61) }
fn main()->int {
 assert (== saved_header 41)
 assert (== (read_header) 41)
 assert (== (local_header) 83)
 assert (== (parameter_header 29) 29)
 assert (== (captured_header) 61)
 assert (== HEADER_NEGATIVE -7)
 assert (== HEADER_HEX 4660)
 assert (== HEADER_OCTAL 10)
 assert (== HEADER_SUFFIX 73)
 assert (== HEADER_REVERSE_SUFFIX 23)
 assert (== HEADER_ZERO 0)
 assert (== HEADER_HEX_MAX 9223372036854775807)
 assert (== HEADER_COMMENT 19)
 assert (== HEADER_MAX 9223372036854775807)
 assert (== HEADER_MIN (- -9223372036854775807 1))
 assert (== (values.declared) 17)
 return 0
}
shadow main { assert (== (main) 0) }
''')
            environment = {**os.environ, 'NANO_BUILD_CACHE': str(work / 'cache'),
                           'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'}
            module = work / 'out.nvm'
            self.paired(source, work)

            for body in ('set HEADER_VALUE 0', 'let missing:int = HEADER_UNDEFINED', 'let invalid:int = (HEADER_VALUE)'):
                with self.subTest(refusal=body):
                    source.write_text(f'module "{dependency}" as values\n'
                                      + 'fn main()->int { ' + body + ' return 0 }\n'
                                      + 'shadow main { assert true }\n')
                    module.write_bytes(b'prior output\n')
                    result = subprocess.run(
                        [str(ROOT / 'bin' / self.compiler), str(source), '--emit-nvm', '-o', str(module)],
                        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('type check failed' if self.compiler == 'nano_virt' else 'NSType checking failed', result.stdout + result.stderr)
                    self.assertEqual(module.read_bytes(), b'prior output\n')

    def test_noninteger_or_out_of_range_header_values_are_not_imported(self):
        cases = ('3.5', '1e3', '1 << 4', '1 + 2', '08', '0x1p2', '2oops',
                 '9223372036854775808', '-9223372036854775809',
                 '0xffffffffffffffffULL', '-1U', '2ulL', '1UU', '1LLL',
                 '1 /* note */ + 2', '1 /* unfinished',
                 '1' + ' ' * 1100 + '+ 2', '1 \\\n + 2')
        for value in cases:
            with self.subTest(value=value), tempfile.TemporaryDirectory(prefix='nano-header-refusal-') as temporary:
                work = Path(temporary).resolve()
                dependency = private_header(work, '#define INVALID_HEADER ' + value + '\n')
                source = work / 'main.nano'
                source.write_text(f'module "{dependency}" as values\n'
                                  'fn main()->int { return INVALID_HEADER }\n'
                                  'shadow main { assert true }\n')
                output = work / 'prior.nvm'
                output.write_bytes(b'prior output\n')
                result = subprocess.run(
                    [str(ROOT / 'bin' / self.compiler), str(source), '--emit-nvm', '-o', str(output)],
                    cwd=ROOT, capture_output=True, text=True, timeout=120,
                    env={**os.environ, 'NANO_BUILD_CACHE': str(work / 'cache')})
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('type check failed' if self.compiler == 'nano_virt' else 'NSType checking failed', result.stdout + result.stderr)
                self.assertEqual(output.read_bytes(), b'prior output\n')


if __name__ == '__main__':
    unittest.main()
