"""I qualify the canonical Builder family without replacing original source assertions."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import unittest
from tests import nanolang_translator_routes as routes
from tests import test_native_regex_host_abi as abi

ROOT = abi.ROOT
ORIGINALS = ['test_std_collections.nano', 'test_std_lib_collections.nano', 'test_import_aliasing.nano']
BUILDER_SOURCE = '''from "std/collections/stringbuilder.nano" import StringBuilder, NLStringBuilder, sb_new, sb_with_capacity, sb_append, sb_append_int, sb_append_char, sb_append_line, sb_to_string, sb_length, sb_capacity, sb_clear, sb_is_empty, sb_free, sb_from_parts, sb_join, sb_repeat, sb_indent
struct Box { item: StringBuilder }
fn builder_wrap(raw:NLStringBuilder)->StringBuilder{return StringBuilder{raw:raw}}
shadow builder_wrap{
 let absent:StringBuilder=(builder_wrap 0)
 assert (== (sb_to_string absent) "") (sb_free absent)
}
fn builder_preserve(value: StringBuilder)->StringBuilder{return value}
shadow builder_preserve{
 let b:StringBuilder=(sb_new)
 let kept:StringBuilder=(builder_preserve (sb_append b "kept"))
 assert (== (sb_to_string kept) "kept") (sb_free b)
}
fn builder_select(left:bool, first:StringBuilder, second:StringBuilder)->StringBuilder{
 return if left {first} else {second}
}
shadow builder_select{
 let a:StringBuilder=(sb_new) let b:StringBuilder=(sb_new)
 let kept:StringBuilder=(sb_append (builder_select true a b) "a")
 assert (== (sb_length a) 1) assert (== (sb_length b) 0)
 (sb_free a) (sb_free b)
}
fn main()->int{
 let builder:StringBuilder=(sb_new)
 assert (== (sb_capacity builder) 256) assert (sb_is_empty builder)
 let alias:StringBuilder=(sb_append builder "alpha")
 let numbered:StringBuilder=(sb_append_int alias 7)
 let marked:StringBuilder=(sb_append_char numbered 33)
 let line:StringBuilder=(sb_append_line marked "")
 assert (== (sb_length line) 8)
 let snapshot:string=(sb_to_string builder)
 assert (== snapshot "alpha7!\\n")
 let stored:array<Box> = [Box{item:builder}]
 let box:Box=(at stored 0)
 let callback:fn(StringBuilder)->StringBuilder=builder_preserve
 let forwarded:StringBuilder=(callback box.item)
 let cleared:StringBuilder=(sb_clear forwarded)
 assert (sb_is_empty alias) assert (== (sb_to_string cleared) "")
 let fresh:StringBuilder=(sb_with_capacity 17)
 assert (== (sb_capacity fresh) 17)
 let grown:StringBuilder=(sb_append fresh "abcdefghijklmnopqrstuvwxyz")
 assert (== (sb_length grown) 26) assert (>= (sb_capacity grown) 27)
 let chosen:StringBuilder=(builder_select false builder grown)
 assert (== (sb_to_string chosen) "abcdefghijklmnopqrstuvwxyz")
 (sb_free alias) (sb_free builder)
 assert (== (sb_length forwarded) 0) assert (== (sb_capacity cleared) 0)
 assert (== (sb_to_string (sb_append forwarded "ignored")) "")
 assert (== (sb_to_string (sb_append_char forwarded 65)) "")
 assert (== snapshot "alpha7!\\n")
 let grown_snapshot:string=(sb_to_string grown)
 (sb_free fresh) assert (== grown_snapshot "abcdefghijklmnopqrstuvwxyz")
 let minimum:StringBuilder=(sb_with_capacity -1)
 assert (== (sb_capacity minimum) 16) (sb_free minimum)
 let absent:StringBuilder=(builder_wrap 0)
 assert (== (sb_length absent) 0) assert (== (sb_capacity absent) 0)
 assert (== (sb_to_string (sb_append absent "ignored")) "")
 assert (== (sb_to_string (sb_append_char absent 65)) "")
 assert (sb_is_empty (sb_clear absent)) (sb_free absent)
 let mut saved:array<string> = [] let mut index:int=0
 while (< index 32){
  let b:StringBuilder=(sb_new)
  let value:StringBuilder=(sb_append b "v")
  set saved (array_push saved (sb_to_string value))
  (sb_free b) set index (+ index 1)
 }
 assert (== (array_length saved) 32) assert (== (at saved 0) "v") assert (== (at saved 31) "v")
 assert (== (sb_from_parts ["a", "b"]) "ab")
 assert (== (sb_join ["a", "b"] ",") "a,b")
 assert (== (sb_repeat "xy" 3) "xyxyxy") assert (== (sb_repeat "x" 0) "")
 assert (== (sb_indent 2 3) "      ")
 return 0
}
shadow main{assert (== (main) 0)}
'''


def active_originals():
    """I insert exact original shadow statements into an additional active main."""
    choices = [('test_std_collections.nano', ['stringbuilder_basic', 'hashmap_basic', 'set_basic']),
               ('test_std_lib_collections.nano', ['main']),
               ('test_import_aliasing.nano', ['joined_alias', 'joined_module'])]
    for name, shadows in choices:
        original = (ROOT/'tests'/name).read_text()
        bodies = []
        for shadow in shadows:
            marker = 'shadow '+shadow+' {'
            assert original.count(marker) == 1
            body = original.split(marker, 1)[1].split('\n}', 1)[0]
            # These retained originals have no nested braced statements in their
            # selected shadow bodies. I refuse a changed shape rather than trim it.
            assert '{' not in body and '}' not in body
            bodies.append('if true {'+body+'\n}\n')
        header = 'fn main() -> int {'
        assert original.count(header) == 1
        yield name, original.replace(header, header+'\n'+''.join(bodies), 1)


class BuilderSourceRoutes(unittest.TestCase):
    command = abi.NativeRegexHostAbi.command
    directory = abi.NativeRegexHostAbi.directory
    exercise_source = abi.NativeRegexHostAbi.exercise_source

    def test_original_collection_and_alias_sources(self):
        for name in ORIGINALS:
            with self.subTest(source=name):
                source = ROOT/'tests'/name
                before = hashlib.sha256(source.read_bytes()).hexdigest()
                self.exercise_source(source.read_text())
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_every_original_collection_assertion_in_an_active_main(self):
        for name, source in active_originals():
            with self.subTest(source=name):
                self.exercise_source(source)

    def test_complete_builder_family_alias_and_owned_snapshots(self):
        self.exercise_source(BUILDER_SOURCE)

    def test_declared_builder_spellings_are_ordinary_functions(self):
        contracts = [
            ('nl_sb_new', '', '', '17', 17),
            ('nl_sb_with_capacity', 'n:int', '7', '(+ n 1)', 8),
            ('nl_sb_append', 'a:int,b:int', '7 2', '(+ a b)', 9),
            ('nl_sb_append_char', 'a:int,b:int', '7 2', '(- a b)', 5),
            ('nl_sb_clear', 'n:int', '7', '(+ n 2)', 9),
            ('nl_sb_length', 'n:int', '7', '(+ n 3)', 10),
            ('nl_sb_capacity', 'n:int', '7', '(+ n 4)', 11),
            ('nl_sb_to_string', 'n:int', '7', '(+ n 5)', 12),
            ('nl_sb_free', 'n:int', '7', '(+ n 6)', 13)]
        source = ''
        checks = []
        for name, parameters, arguments, result, expected in contracts:
            call = '('+name+(' '+arguments if arguments else '')+')'
            check = 'assert (== '+call+' '+str(expected)+')'
            source += 'fn '+name+'('+parameters+')->int{return '+result+'}\n'
            source += 'shadow '+name+'{'+check+'}\n'
            checks.append(check)
        source += 'fn main()->int{'+ ' '.join(checks)+' return 0}\nshadow main{assert (== (main) 0)}\n'
        self.assertEqual(len(contracts), 9)
        self.exercise_source(source)

    def test_canonical_provider_descriptors_and_relocated_compilation(self):
        pure = routes.source_compiler_command()
        self.assertTrue(pure)
        names = ['new', 'with_capacity', 'append', 'append_char', 'clear', 'length', 'capacity', 'to_string', 'free']
        for label, compiler in [('c', [ROOT/'bin/nano_virt']), ('pure', pure)]:
            with self.subTest(producer=label):
                work = self.directory('builder-descriptor-'+label+'-')
                source, module = work/'main.nano', work/'main.nvm'
                source.write_text(BUILDER_SOURCE)
                self.command(work, 'compile', [*compiler, source, '--emit-nvm', '-o', module], build=True)
                dump = self.command(work, 'dump', [ROOT/'bin/nanoisa', 'dump', module]).stdout
                descriptors = re.findall(r'(?m)^\.import ("[^"\n]*") ("nl_sb_[^"\n]*")[^\n]*$', dump)
                self.assertEqual({json.loads(name) for _, name in descriptors}, {'nl_sb_'+name for name in names})
                providers = {json.loads(path) for path, _ in descriptors}
                self.assertEqual(len(providers), 1)
                provider = Path(providers.pop())
                self.assertTrue(provider.is_absolute() and provider.is_file())
                self.assertRegex(provider.name, r'^libstd_collections\.(so|dylib)$')
                self.assertRegex(dump, r'(?m)^\.import_kind \d+ artifact$')
                import shlex
                imported = [shlex.split(line) for line in dump.splitlines() if line.startswith('.import ')]
                for index, entry in enumerate(imported):
                    if entry[2] in {'nl_sb_'+name for name in names}:
                        self.assertRegex(dump, r'(?m)^\.import_kind '+str(index)+r' artifact$')
                self.assertNotIn('.import "" "nl_sb_', dump)
                if label == 'pure':
                    from tests.native_process import run_captured
                    env = dict(os.environ)
                    env.pop('NANOLANG_ROOT', None)
                    relocated = work/'relocated.nvm'
                    result = run_captured([*compiler, source, '--emit-nvm', '-o', relocated],
                                          cwd=work, text=True, timeout=60, build=True, env=env)
                    (work/'relocated.stdout').write_text(result.stdout)
                    (work/'relocated.stderr').write_text(result.stderr)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    self.command(work, 'relocated-vm', [ROOT/'bin/nano_vm', relocated])

    def test_cross_kind_record_values_preserve_prior_output(self):
        pure = routes.source_compiler_command()
        self.assertTrue(pure)
        imports = ('from "std/collections/stringbuilder.nano" import StringBuilder,sb_new,sb_append\n'
                   'from "std/collections/hashmap.nano" import HashMap,map_new\n'
                   'from "std/collections/set.nano" import Set,set_new\n')
        prefix = 'let b:StringBuilder=(sb_new) let hm:HashMap=(map_new) let members:Set=(set_new) '
        bodies = ['let wrong:StringBuilder=StringBuilder{raw:hm.raw}',
                  'let wrong:StringBuilder=StringBuilder{raw:members.raw}',
                  'let wrong:HashMap=HashMap{raw:b.raw}',
                  'let wrong:Set=Set{raw:b.raw}',
                  'let wrong:StringBuilder=StringBuilder{raw:if true {b.raw} else {hm.raw}}',
                  'let wrong:StringBuilder=StringBuilder{raw:if false {members.raw} else {b.raw}}',
                  'let wrong:StringBuilder=StringBuilder{raw:(cond (true b.raw) (else hm.raw))}',
                  'let wrong:StringBuilder=StringBuilder{raw:0}',
                  'let wrong:StringBuilder=StringBuilder{raw:7}',
                  'let wrong:StringBuilder=(sb_append b 7)']
        for label, compiler in [('c', [ROOT/'bin/nano_virt']), ('pure', pure)]:
            for index, body in enumerate(bodies):
                with self.subTest(producer=label, case=index):
                    work = self.directory('builder-kind-refusal-'+label+'-')
                    source, module = work/'main.nano', work/'prior.nvm'
                    source.write_text(imports+'fn main()->int{'+prefix+body+' return 0}\nshadow main{assert true}\n')
                    prior = b'I retain my prior Builder kind output.\n'
                    module.write_bytes(prior)
                    result = self.command(work, 'refusal', [*compiler, source, '--emit-nvm', '-o', module], build=True, accepted=False)
                    self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(module.read_bytes(), prior)

    def test_noncanonical_builder_copy_preserves_prior_output(self):
        pure = routes.source_compiler_command()
        self.assertTrue(pure)
        work = self.directory('builder-noncanonical-')
        copied = work/'copy'
        copied.mkdir()
        for name in ['stringbuilder.nano', 'collections.c', 'collections.h', 'module.json']:
            shutil.copy2(ROOT/'modules/std/collections'/name, copied/name)
        source, module = work/'main.nano', work/'prior.nvm'
        source.write_text('from '+json.dumps(str(copied/'stringbuilder.nano'))+' import StringBuilder,sb_new,sb_free\n'
                          'fn main()->int{let b:StringBuilder=(sb_new) (sb_free b) return 0}\n'
                          'shadow main{assert (== (main) 0)}\n')
        prior = b'I retain my prior noncanonical Builder output.\n'
        module.write_bytes(prior)
        result = self.command(work, 'refusal', [*pure, source, '--emit-nvm', '-o', module], build=True, accepted=False)
        self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(module.read_bytes(), prior)
        self.assertIn('StringBuilder', result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
