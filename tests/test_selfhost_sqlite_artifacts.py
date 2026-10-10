"""I execute SQLite's exact artifact ABI through checked VM and native routes."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '''unsafe module "modules/sqlite/sqlite.nano" as sqlite
fn main()->int {
 assert (> (str_length (sqlite.version)) 0)
 assert (> (sqlite.version_number) 0)
 let db:int = (sqlite.open ":memory:")
 assert (!= db 0)
 assert (== (sqlite.exec db "CREATE TABLE data (id INTEGER, value REAL, text TEXT, missing TEXT)") 0)
 assert (== (sqlite.begin_transaction db) 0)
 let insert:int = (sqlite.prepare db "INSERT INTO data VALUES (?, ?, ?, ?)")
 assert (!= insert 0)
 assert (== (sqlite.bind_int insert 1 42) 0)
 assert (== (sqlite.bind_double insert 2 3.5) 0)
 assert (== (sqlite.bind_text insert 3 "retained text") 0)
 assert (== (sqlite.bind_null insert 4) 0)
 assert (== (sqlite.step insert) (sqlite.SQLITE_DONE))
 assert (== (sqlite.changes db) 1)
 assert (== (sqlite.last_insert_rowid db) 1)
 assert (== (sqlite.reset insert) 0)
 assert (== (sqlite.finalize insert) 0)
 assert (== (sqlite.commit db) 0)
 assert (== (sqlite.begin_transaction db) 0)
 assert (== (sqlite.exec db "DELETE FROM data") 0)
 assert (== (sqlite.rollback db) 0)
 let query:int = (sqlite.prepare db "SELECT id, value, text, missing FROM data")
 assert (!= query 0)
 assert (== (sqlite.column_count query) 4)
 assert (== (sqlite.step query) (sqlite.SQLITE_ROW))
 assert (== (sqlite.column_int query 0) 42)
 assert (== (sqlite.column_double query 1) 3.5)
 assert (== (sqlite.column_type query 3) (sqlite.SQLITE_NULL))
 let name:string = (sqlite.column_name query 2)
 let text:string = (sqlite.column_text query 2)
 assert (== (sqlite.column_text query 3) "")
 assert (!= (sqlite.close db) 0)
 assert (== (sqlite.finalize query) 0)
 assert (== (sqlite.prepare db "not valid SQL") 0)
 let error:string = (sqlite.errmsg db)
 assert (str_contains error "syntax error")
 assert (== (sqlite.exec db "DELETE FROM data") 0)
 assert (== (sqlite.close db) 0)
 assert (== name "text")
 assert (== text "retained text")
 assert (str_contains error "syntax error")
 assert (!= (sqlite.close 0) 0)
 assert (!= (sqlite.finalize 0) 0)
 assert (== (sqlite.column_text 0 0) "")
 assert (== (sqlite.errmsg 0) "Invalid database handle")
 return 0
}
shadow main { assert true }
'''


class SqliteArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-sqlite-artifacts-'))
        print('I retain SQLite artifact evidence at', cls.work, flush=True)
        cls.sequence = 0
        cls.env = {**os.environ, 'NANO_BUILD_CACHE': str(cls.work / 'cache'),
                   'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1'}
        cls.cc = shlex.split(os.environ.get('NANO_NATIVE_TEST_CC') or
                            os.environ.get('CC') or 'cc')

    @classmethod
    def command(cls, *args, success=True):
        cls.sequence += 1
        command = list(map(str, args))
        result = subprocess.run(command, cwd=ROOT, env=cls.env, capture_output=True,
                                text=True, timeout=180)
        (cls.work / f'command-{cls.sequence}.log').write_text(
            repr(command) + '\n' + result.stdout + result.stderr + f'\nexit={result.returncode}\n')
        if (result.returncode == 0) != success:
            raise AssertionError(f'{command}: {result.returncode}\n{result.stdout}\n{result.stderr}')
        return result

    def native(self, module):
        source, binary = module.with_suffix('.c'), module.with_suffix('.native')
        self.command(ROOT / 'bin/nvm2c', module, '-o', source)
        self.command(*self.cc, '-std=c11', '-Wall', '-Wextra', '-Werror',
                     '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                     source, ROOT / 'bin/nano_aot_runtime.o',
                     *shlex.split(os.environ.get('LDFLAGS', '')),
                     *(['-Wl,--export-dynamic'] if sys.platform.startswith('linux') else []),
                     '-ldl', '-lm', '-o', binary)
        return binary

    def paired(self, source, compiler, expected):
        module = self.work / (source.stem + '-' + compiler.name + '.nvm')
        self.command(compiler, source, '--emit-nvm', '-o', module)
        self.command(ROOT / 'bin/nano_vm', '--verify-only', module)
        self.assertEqual(self.command(ROOT / 'bin/nano_vm', module).stdout, expected)
        self.assertEqual(self.command(self.native(module)).stdout, expected)

    def cases(self, compilers):
        source = self.work / 'lifetimes.nano'
        source.write_text(SOURCE)
        for compiler in compilers:
            for path, expected in ((source, ''), (ROOT / 'examples/advanced/sqlite_simple.nano',
                                                 'prepared CRUD: Alice -> Alicia -> deleted\n')):
                with self.subTest(compiler=compiler.name, source=path.name):
                    self.paired(path, compiler, expected)

    def test_seed_real_sqlite_and_borrowed_strings(self):
        self.cases([ROOT / 'bin/nano_virt'])

    def test_selfhost_real_sqlite_and_borrowed_strings(self):
        self.cases([ROOT / 'bin/nanoc_stage1', ROOT / 'bin/nanoc_stage2'])

    def assembly(self, text):
        source, module = self.work / 'input.nasm', self.work / 'input.nvm'
        source.write_text(text)
        self.command(ROOT / 'bin/nanoisa', 'asm', source, '-o', module)
        self.command(ROOT / 'bin/nano_vm', '--verify-only', module)
        return module

    def test_native_refuses_wrong_signature_kind_path_and_symbol(self):
        for path, signature, kind in (
            ('/missing/sqlite.so', '"nl_sqlite3_bind_double" int int int int', 'artifact'),
            ('/missing/sqlite.so', '"nl_sqlite3_column_double" int int int', 'artifact'),
            ('/missing/sqlite.so', '"nl_sqlite3_open" int', 'artifact'),
            ('/missing/sqlite.so', '"nl_sqlite3_unknown" int', 'artifact'),
            ('', '"nl_sqlite3_version" string', 'ffi'),
        ):
            with self.subTest(signature=signature, kind=kind, path=path):
                module = self.assembly(f'.import {json.dumps(path)} {signature}\n'
                                      f'.import_kind 0 {kind}\n.entry main\n'
                                      '.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
                output = self.work / 'refused.c'
                output.write_bytes(b'prior output\n')
                self.command(ROOT / 'bin/nvm2c', module, '-o', output, success=False)
                self.assertEqual(output.read_bytes(), b'prior output\n')

    def test_verifier_refuses_relative_artifact_path(self):
        source, output = self.work / 'relative.nasm', self.work / 'relative.nvm'
        source.write_text('.import "relative.so" "nl_sqlite3_version" string\n'
                          '.import_kind 0 artifact\n.entry main\n'
                          '.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
        output.write_bytes(b'prior output\n')
        result = self.command(ROOT / 'bin/nanoisa', 'asm', source, '-o', output, success=False)
        self.assertIn('artifact path must be absolute', result.stderr)
        self.assertEqual(output.read_bytes(), b'prior output\n')

    def test_selfhost_refuses_wrong_abi_and_preserves_output(self):
        for compiler in (ROOT / 'bin/nanoc_stage1', ROOT / 'bin/nanoc_stage2'):
            for declaration, call in (
                ('extern fn nl_sqlite3_bind_double(a:int,b:int,c:int)->int', '(nl_sqlite3_bind_double 0 0 0)'),
                ('extern fn nl_sqlite3_column_double(a:int,b:int)->int', '(nl_sqlite3_column_double 0 0)'),
                ('extern fn nl_sqlite3_open()->int', '(nl_sqlite3_open)'),
            ):
                with self.subTest(compiler=compiler.name, declaration=declaration):
                    source, output = self.work / 'wrong.nano', self.work / 'wrong.nvm'
                    source.write_text(declaration + '\nfn main()->int { unsafe { let result:int = ' +
                                      call + ' } return 0 }\nshadow main { assert true }\n')
                    output.write_bytes(b'prior output\n')
                    result = self.command(compiler, source, '--emit-nvm', '-o', output, success=False)
                    diagnostic = ('I require exact operands for this artifact call'
                                  if 'bind_double' in declaration else
                                  'I require the exact artifact result and arity')
                    self.assertIn(diagnostic, result.stdout + result.stderr)
                    self.assertEqual(output.read_bytes(), b'prior output\n')

    def test_unused_typed_imports_compile_without_opening_library(self):
        for signature in ('"nl_sqlite3_version" string', '"nl_sqlite3_version_number" int',
                          '"nl_json_parse" opaque string', '"nl_json_object_keys" array opaque'):
            with self.subTest(signature=signature):
                module = self.assembly('.import "/missing/artifact.so" ' + signature +
                                       '\n.import_kind 0 artifact\n.entry main\n'
                                       '.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
                self.command(self.native(module))

    def test_same_symbol_retains_artifact_owner(self):
        libraries = []
        for name in ('left', 'right'):
            source, library = self.work / (name + '.c'), self.work / (name + '.so')
            source.write_text('const char *nl_sqlite3_version(void) { return "' + name + '"; }\n')
            self.command(*self.cc, '-shared', '-fPIC', source, '-o', library)
            libraries.append(library)
        imports = ''.join(f'.import {json.dumps(str(lib))} "nl_sqlite3_version" string\n'
                          f'.import_kind {i} artifact\n' for i, lib in enumerate(libraries))
        module = self.assembly(imports + '.string left "left"\n.string right "right"\n'
                               '.entry main\n.function main 0 1 0 int 1\n'
                               'CALL_EXTERN 0\nSTORE_LOCAL 0\nCALL_EXTERN 1\nPUSH_STR right\nEQ\nASSERT\n'
                               'LOAD_LOCAL 0\nPUSH_STR left\nEQ\nASSERT\nPUSH_I64 0\nRET\n.end\n')
        self.command(ROOT / 'bin/nano_vm', module)
        self.command(self.native(module))
        libraries[0].unlink()
        self.command(self.work / 'input.native', success=False)


if __name__ == '__main__':
    unittest.main()
