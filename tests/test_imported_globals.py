"""I qualify module-owned public global values through self-hosted products."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ImportedGlobals(unittest.TestCase):
    def check(self, source, bad=False, files=None):
        override = os.environ.get('NANO_IMPORTED_GLOBAL_COMPILER')
        compilers = [shlex.split(override)] if override else [[str(ROOT/'bin'/stage)] for stage in ('nanoc_stage1', 'nanoc_stage2')]
        for compiler in compilers:
            with self.subTest(compiler=compiler), tempfile.TemporaryDirectory(prefix='nano-imported-global-') as tmp:
                work = Path(tmp)
                (work/'a.nano').write_text('pub let answer: int = 41\nlet hidden: int = 99\npub let mut count: int = 0\npub fn read_answer()->int{return answer}\nshadow read_answer {assert (== (read_answer) 41)}\n')
                (work/'b.nano').write_text('pub let answer: string = "second"\npub fn read_answer()->string{return answer}\nshadow read_answer {assert (== (read_answer) "second")}\n')
                for name, text in (files or {}).items(): (work/name).write_text(text)
                path, module, native_source, binary = [work/name for name in ('main.nano','main.nvm','main.c','main')]
                path.write_text(source)
                module.write_bytes(b'prior-output')
                env = {**os.environ, 'NANO_BUILD_CACHE': str(work/'cache'), 'ASAN_OPTIONS': 'detect_leaks=1'}
                result = subprocess.run(compiler + [str(path),'--emit-nvm','-o',str(module)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
                if bad:
                    self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                    self.assertEqual(module.read_bytes(),b'prior-output')
                    continue
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                cc=shlex.split(os.environ.get('NANO_NATIVE_TEST_CC',os.environ.get('CC','cc')))
                for command in ([ROOT/'bin/nano_vm','--verify-only',module], [ROOT/'bin/nano_vm',module],
                                [os.environ.get('NANO_IMPORTED_GLOBAL_TRANSLATOR', str(ROOT/'bin/nvm2c')),module,'-o',native_source],
                                cc+['-std=c11','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',native_source,ROOT/'bin/nano_aot_runtime.o','-lm','-o',binary], [binary]):
                    result=subprocess.run(list(map(str,command)),cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_qualified_distinct_types_and_dependency_shadows(self):
        self.check('module "a.nano" as first\nmodule "b.nano" as second\nfn main()->int{assert (== first.answer 41) assert (== second.answer "second") assert (== (first.read_answer) 41) return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_selective_aliases_and_lexical_shadowing(self):
        self.check('from "a.nano" import answer as one\nfrom "b.nano" import answer as two\nfn main()->int{assert (== one 41) assert (== two "second") let one: string = "local" assert (== one "local") return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_mutable_aliases_share_storage(self):
        self.check('from "a.nano" import count as first\nfrom "a.nano" import count as second\nfn main()->int{set first 7 assert (== second 7) set second 0 assert (== first 0) return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_wildcard_public_globals(self):
        self.check('from "a.nano" import *\nfn main()->int{assert (== answer 41) return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_relative_spellings_share_canonical_storage(self):
        self.check('from "a.nano" import count as first\nfrom "./a.nano" import count as second\nfn main()->int{set first 7 assert (== second 7) set second 0 assert (== first 0) return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_repeated_alias_keeps_identity(self):
        self.check('from "a.nano" import answer as value\nfrom "./a.nano" import answer as value\nfn main()->int{return (- value 41)}\nshadow main {assert (== (main) 0)}\n')

    def test_conflicting_aliases_preserve_output(self):
        for imports in ['from "a.nano" import answer as value\nfrom "b.nano" import answer as value',
                        'from "a.nano" import *\nfrom "b.nano" import *',
                        'module "a.nano" as same\nmodule "b.nano" as same']:
            with self.subTest(imports=imports):
                self.check(imports+'\nfn main()->int{return 0}\nshadow main {assert true}\n',bad=True)

    def test_qualified_writes_share_alias_storage(self):
        self.check('module "a.nano" as first\nfrom "a.nano" import count as value\nfn main()->int{set first.count 7 assert (== value 7) set value 3 assert (== first.count 3) set first.count 0 return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_qualified_write_refusals_preserve_output(self):
        for assignment in ['set first.answer 3', 'set first.hidden 3', 'set first.absent 3', 'set first.count "wrong"']:
            with self.subTest(assignment=assignment):
                self.check('module "a.nano" as first\nfn main()->int{'+assignment+' return 0}\nshadow main {assert true}\n',bad=True)

    def test_imported_callable_globals(self):
        for imports, call in [('module "a.nano" as first', 'first.apply'), ('from "a.nano" import apply', 'apply')]:
            with self.subTest(imports=imports):
                self.check(imports+'\nfn main()->int{assert (== ('+call+' 4) 5) return 0}\nshadow main {assert (== (main) 0)}\n',
                           files={'a.nano':'pub let apply: fn(int)->int = fn(x:int)->int{return (+ x 1)}\n'})
                for arguments in ['true', '', '1 2']:
                    self.check(imports+'\nfn main()->int{return ('+call+' '+arguments+')}\nshadow main {assert true}\n',bad=True,
                               files={'a.nano':'pub let apply: fn(int)->int = fn(x:int)->int{return (+ x 1)}\n'})

    def test_imported_integer_arrays(self):
        for imports, name in [('module "a.nano" as first', 'first.values'), ('from "a.nano" import values', 'values')]:
            with self.subTest(imports=imports):
                self.check(imports+'\nfn main()->int{assert (== (at '+name+' 1) 7) return 0}\nshadow main {assert (== (main) 0)}\n',
                           files={'a.nano':'pub let values: array<int> = [3,7]\n'})

    def test_diamond_initialization_runs_once(self):
        self.check('module "left.nano" as left\nmodule "right.nano" as right\nmodule "a.nano" as first\nfn main()->int{assert (== (left.read) 1) assert (== (right.read) 1) assert (== first.count 1) return 0}\nshadow main {assert (== (main) 0)}\n',files={
            'a.nano':'pub let mut count: int = 0\nfn next()->int{set count (+ count 1) return count}\nshadow next {let saved: int = count set count 0 assert (== (next) 1) set count saved}\npub let value: int = (next)\n',
            'left.nano':'module "a.nano" as first\npub fn read()->int{return first.value}\nshadow read {assert (== (read) 1)}\n',
            'right.nano':'module "./a.nano" as first\npub fn read()->int{return first.value}\nshadow read {assert (== (read) 1)}\n'})

    def test_imported_qualified_string_array(self):
        self.check('module "a.nano" as first\nfn main()->int{set first.values (array_push first.values "kept") assert (== (at first.values 0) "kept") set first.values [] return 0}\nshadow main {assert (== (main) 0)}', files={'a.nano': 'pub let mut values: array<string> = []\n'})

    def test_imported_selective_string_array(self):
        self.check('from "a.nano" import values\nfn main()->int{set values (array_push values "kept") assert (== (at values 0) "kept") set values [] return 0}\nshadow main {assert (== (main) 0)}', files={'a.nano': 'pub let mut values: array<string> = []\n'})

    def test_imported_qualified_map(self):
        self.check('module "a.nano" as first\nfn main()->int{(map_put first.values "key" "kept") assert (== (map_get first.values "key") "kept") return 0}\nshadow main {assert (== (main) 0)}', files={'a.nano': 'pub let values: HashMap<string,string> = (map_new)\n'})

    def test_imported_selective_map(self):
        self.check('from "a.nano" import values\nfn main()->int{(map_put values "key" "kept") assert (== (map_get values "key") "kept") return 0}\nshadow main {assert (== (main) 0)}', files={'a.nano': 'pub let values: HashMap<string,string> = (map_new)\n'})

    def test_imported_qualified_record(self):
        self.check('module "a.nano" as first\nfn main()->int{assert (== first.value.text "kept") return 0}\nshadow main {assert (== (main) 0)}', files={'a.nano': 'struct Record { text: string }\npub let value: Record = Record { text: "kept" }\n'})

    def test_imported_selective_record(self):
        self.check('from "a.nano" import value\nfn main()->int{assert (== value.text "kept") return 0}\nshadow main {assert (== (main) 0)}', files={'a.nano': 'struct Record { text: string }\npub let value: Record = Record { text: "kept" }\n'})

    def test_imported_record_module_identity(self):
        self.check('module "a.nano" as first\nmodule "b.nano" as second\nfn main()->int{assert (== first.box.value 41) assert (== second.box.value "second") return 0}\nshadow main {assert (== (main) 0)}', files={'a.nano': 'struct Box { value: int }\npub let box: Box = Box { value: 41 }\n', 'b.nano': 'struct Box { value: string }\npub let box: Box = Box { value: "second" }\n'})

    def test_imported_closure_global_capture(self):
        self.check('module "a.nano" as first\nfn main()->int{assert (== (first.apply 4) 11) return 0}\nshadow main {assert (== (main) 0)}', files={'a.nano': 'fn make(base:int)->fn(int)->int{return fn(value:int)->int{return (+ base value)}}\nshadow make {let f:fn(int)->int=(make 3) assert (== (f 4) 7)}\npub let apply:fn(int)->int=(make 7)\n'})

    def test_imported_call_snapshot_before_argument(self):
        self.check('module "a.nano" as first\nfn main()->int{let result:int=(first.apply (first.swap)) assert (== result 2) assert (== (first.apply 1) 101) (first.reset) return 0}\nshadow main {assert (== (main) 0)}', files={'a.nano': 'pub let mut apply:fn(int)->int=fn(x:int)->int{return (+ x 1)}\npub fn swap()->int{set apply fn(x:int)->int{return (+ x 100)} return 1}\nshadow swap {let saved:fn(int)->int=apply assert (== (swap) 1) assert (== (apply 1) 101) set apply saved}\npub fn reset()->void{set apply fn(x:int)->int{return (+ x 1)}}\nshadow reset {(reset) assert (== (apply 1) 2)}\n'})

    def test_qualified_map_type_refusals(self):
        for body in ['(map_put first.values 7 "kept")', '(map_put first.values "key" 7)', 'let first: int = 1 (map_put first.values "key" "kept")']:
            with self.subTest(body=body):
                self.check('module "a.nano" as first\nfn main()->int{'+body+' return 0}\nshadow main {assert true}', bad=True,
                           files={'a.nano':'pub let values: HashMap<string,string> = (map_new)\n'})

    def test_resource_bearing_globals_preserve_output(self):
        for payload in ['Handle', 'array<Handle>']:
            with self.subTest(payload=payload):
                module = ('resource struct Handle { fd: int }\nunion Box<T> { Some { value: T }, None {} }\n'
                          'struct Outer { boxed: Box<'+payload+'> }\npub let owner: Outer = Outer { boxed: Box.None {} }\n')
                self.check('module "a.nano" as first\nfn main()->int{return 0}\nshadow main {assert true}\n',
                           bad=True, files={'a.nano':module})

    def test_immutable_public_value_in_pure_function(self):
        self.check('module "a.nano" as first\npure fn answer()->int{return first.answer}\nshadow answer {assert (== (answer) 41)}\nfn main()->int{assert (== (answer) 41) return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_legacy_plain_import_keeps_immutable_constants(self):
        self.check('import "a.nano"\nfn main()->int{assert (== hidden 99) return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_transitive_selective_import_keeps_declaring_context(self):
        self.check('module "bridge.nano" as bridge\nfn main()->int{assert (== (bridge.read) 41) return 0}\nshadow main {assert (== (main) 0)}\n',files={
            'bridge.nano':'from "a.nano" import answer as selected\npub fn read()->int{return selected}\nshadow read {assert (== (read) 41)}\n'})

    def test_captured_local_shadows_imported_alias(self):
        self.check('from "a.nano" import answer as value\nfn make(value: int)->fn()->int{return fn()->int{return value}}\nshadow make {let read: fn()->int = (make 7) assert (== (read) 7)}\nfn main()->int{let read: fn()->int = (make 13) assert (== (read) 13) return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_private_missing_wrong_type_and_immutable_write_preserve_output(self):
        for source in ['module "a.nano" as first\nfn main()->int{return first.hidden}',
                       'from "a.nano" import hidden as value\nfn main()->int{return value}',
                       'from "a.nano" import absent as value\nfn main()->int{return value}',
                       'from "a.nano" import answer\nfn main()->int{let value: string = answer return 0}',
                       'from "a.nano" import answer\nfn main()->int{set answer 3 return 0}',
                       'module "a.nano" as first\npure fn read()->int{return first.count}\nshadow read {assert true}\nfn main()->int{return 0}']:
            with self.subTest(source=source):
                self.check(source+'\nshadow main {assert true}\n',bad=True)


if __name__ == '__main__':
    unittest.main()
