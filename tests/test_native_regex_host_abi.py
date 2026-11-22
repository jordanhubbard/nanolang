"""I keep regex native admission on exact builtin signatures and real source routes."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch
from tests import nanolang_translator_routes as routes
from tests import test_numeric_builtin_selection as numeric
from tests.native_process import native_link_flags

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = [
    ('nl_regex_compile', 'opaque', ['string']),
    ('nl_regex_match', 'int', ['opaque', 'string']),
    ('nl_regex_find', 'int', ['opaque', 'string']),
    ('nl_regex_find_all', 'array', ['opaque', 'string']),
    ('nl_regex_groups', 'array', ['opaque', 'string']),
    ('nl_regex_replace', 'string', ['opaque', 'string', 'string']),
    ('nl_regex_replace_all', 'string', ['opaque', 'string', 'string']),
    ('nl_regex_split', 'array', ['opaque', 'string']),
    ('nl_regex_free', 'void', ['opaque']),
]


class NativeRegexHostAbi(unittest.TestCase):
    command = numeric.NumericBuiltinSelection.command

    def directory(self, prefix):
        output = ROOT / '.test_output/native-regex-host-abi'
        output.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=output))

    def native(self, work, module):
        generated, native = work / 'program.c', work / 'program'
        self.command(work, 'translate', [ROOT/'bin/nvm2c', module, '-o', generated], build=True)
        self.command(work, 'native-build', [*shlex.split(os.environ.get('NANO_NATIVE_TEST_CC', 'cc')),
            '-std=c11', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
            generated, ROOT/'bin/nano_aot_runtime.o', '-lm', *native_link_flags(), '-o', native], build=True)
        result = self.command(work, 'native-run', [native])
        self.assertEqual((result.stdout, result.stderr), ('', ''))

    def test_wrong_namespace_kind_and_signatures_preserve_output(self):
        cases = []
        for name, result, parameters in CONTRACTS:
            cases.append(('', name, 'string' if result == 'int' else 'int', parameters, None))
            for index in range(len(parameters)):
                wrong = list(parameters)
                wrong[index] = 'int' if wrong[index] != 'int' else 'string'
                cases.append(('', name, result, wrong, None))
            cases.append(('', name, result, parameters[:-1], None))
            cases.append(('libregex.so', name, result, parameters, None))
            cases.append(('./std/regex/regex.nano', name, result, parameters, None))
            cases.append(('/tmp/regex-provider.so', name, result, parameters, 'artifact'))
            cases.append(('', name, result, parameters, 'coprocess'))
        cases.append(('', 'nl_regex_other', 'int', ['string'], None))
        for index, (library, name, result, parameters, kind) in enumerate(cases):
            with self.subTest(library=library, name=name, result=result, parameters=parameters, kind=kind):
                work = self.directory('refusal-')
                assembly, module, generated = work/'program.nasm', work/'program.nvm', work/'prior.c'
                assembly.write_text('.entry main\n.import '+json.dumps(library)+' '+json.dumps(name)+' '+result+
                    (' '+ ' '.join(parameters) if parameters else '')+'\n'+
                    ('.import_kind 0 '+kind+'\n' if kind else '')+
                    '.function main 0 0 0 int 1\nPUSH_I64 0\nRET\n.end\n')
                self.command(work, 'assemble', [ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
                self.command(work, 'verify', [ROOT/'bin/nano_vm', '--verify-only', module])
                prior = b'I preserve my prior regex translation.\n'
                generated.write_bytes(prior)
                observed = self.command(work, 'refuse', [ROOT/'bin/nvm2c', module, '-o', generated], accepted=False)
                self.assertGreater(observed.returncode, 0, observed.stdout+observed.stderr)
                self.assertNotRegex(observed.stdout+observed.stderr, r'AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:')
                self.assertEqual(generated.read_bytes(), prior)
        self.assertEqual(len(cases), 73)

    def test_duplicate_exact_builtin_imports_execute(self):
        work = self.directory('duplicate-')
        assembly, module = work/'program.nasm', work/'program.nvm'
        assembly.write_text('''.entry main
.import "" "nl_regex_compile" opaque string
.import "" "nl_regex_free" void opaque
.import "" "nl_regex_compile" opaque string
.string pattern "a"
.function main 0 0 0 int 1
PUSH_STR pattern
CALL_EXTERN 2
CALL_EXTERN 1
PUSH_I64 0
RET
.end
''')
        self.command(work, 'assemble', [ROOT/'bin/nanoisa', 'asm', assembly, '-o', module])
        self.command(work, 'verify', [ROOT/'bin/nano_vm', '--verify-only', module])
        self.native(work, module)

    def exercise_source(self, source, expected=''):
        work = self.directory('source-')
        fixture = work/'main.nano'
        fixture.write_text(source)
        environment = {}
        if not routes.source_compiler_command():
            provider = ROOT / 'bin/nanoc_stage1'
            self.assertTrue(provider.is_file(), 'I require an actual Stage1 or explicitly selected pure compiler product.')
            environment['NANOLANG_TRANSLATOR_SOURCE_COMPILER'] = str(provider)
        names = ['bin/nanoc_c', 'bin/nano_virt', 'bin/nano_vm', 'bin/nvm2c', 'bin/nano_aot_runtime.o']
        before = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
        original = routes.run_captured
        count = 0
        def observed(command, **kwargs):
            nonlocal count
            result = original(command, **kwargs)
            (work/(str(count)+'.stdout')).write_bytes(result.stdout)
            (work/(str(count)+'.stderr')).write_bytes(result.stderr)
            with (work/'commands.jsonl').open('a') as stream:
                stream.write(json.dumps({'index': count, 'argv': list(map(str, command)),
                    'returncode': result.returncode, 'build': kwargs.get('build',False),
                    'timeout': kwargs.get('timeout')})+'\n')
            count += 1
            self.assertNotRegex((result.stdout+result.stderr).decode(errors='replace'),
                r'AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:')
            return result
        try:
            with patch.dict(os.environ, environment), patch.object(routes, 'run_captured', observed):
                products = routes.build_routes(fixture, work)
                self.assertEqual(len(products), 5)
                for label, command in products:
                    with self.subTest(route=label): self.assertEqual(routes.checked(command,timeout=30),expected)
        finally:
            after = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
            (work/'inputs.json').write_text(json.dumps({'before':before,'after':after,
                'source_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest()},indent=2)+'\n')
            self.assertEqual(before, after)

    def test_original_split_and_groups_all_source_routes(self):
        for name, expected in [('test_split2.nano', 'Parts: 3\nPart 0: a\nPart 1: b\nPart 2: c\n'),
                               ('test_std_regex_groups_uaf.nano', '')]:
            with self.subTest(source=name): self.exercise_source((ROOT/'tests'/name).read_text(),expected)

    def test_complete_regex_family_and_result_lifetimes(self):
        self.exercise_source('''from "std/regex/regex.nano" import Regex, compile, matches, find, find_all, groups, replace, replace_all, split, free, is_null
fn regex_compile()->int{return 17}
shadow regex_compile{assert (== (regex_compile) 17)}
fn main()->int{
 assert (== (regex_compile) 17)
 let re: Regex = (compile "(a)(b)")
 assert (not (is_null re))
 assert (matches re "zabz")
 assert (== (find re "zabz") 1)
 let positions: array<int> = (find_all re "ab ab")
 assert (== (array_length positions) 2)
 assert (== (at positions 0) 0)
 assert (== (at positions 1) 3)
 let captures: array<string> = (groups re "ab")
 let first: string = (replace re "ab ab" "long")
 let all: string = (replace_all re "ab ab" "long")
 let pieces: array<string> = (split re "leftabright")
 (free re)
 assert (== (at captures 0) "ab")
 assert (== (at captures 1) "a")
 assert (== (at captures 2) "b")
 assert (== first "long ab")
 assert (== all "long long")
 assert (== (array_length pieces) 2)
 assert (== (at pieces 0) "left")
 assert (== (at pieces 1) "right")
 let invalid: Regex = (compile "[")
 assert (is_null invalid)
 assert (not (matches invalid "text"))
 assert (== (find invalid "text") -1)
 assert (== (array_length (find_all invalid "text")) 0)
 assert (== (array_length (groups invalid "text")) 0)
 assert (== (replace invalid "text" "long") "text")
 assert (== (replace_all invalid "text" "long") "text")
 assert (== (array_length (split invalid "text")) 0)
 (free invalid)
 return 0
}
shadow main{assert (== (main) 0)}
''')
