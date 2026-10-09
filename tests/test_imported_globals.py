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
                                [ROOT/'bin/nvm2c',module,'-o',native_source],
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
