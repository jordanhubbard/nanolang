"""I preserve declaration priority and lexical bindings in checked bytecode."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADERS = [Path(p) / 'sqlite3.h' for p in
           ('/opt/homebrew/include', '/usr/local/include', '/usr/include')]


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


if __name__ == '__main__':
    unittest.main()
