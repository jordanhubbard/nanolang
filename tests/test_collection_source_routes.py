"""I keep checked collection handles bound to canonical declarations in source products."""
import hashlib
import json
import os
import unittest
from pathlib import Path

from tests import test_native_regex_host_abi as abi
from tests import nanolang_translator_routes as routes

ROOT = abi.ROOT


class CollectionSourceRoutes(unittest.TestCase):
    command = abi.NativeRegexHostAbi.command
    directory = abi.NativeRegexHostAbi.directory
    exercise_source = abi.NativeRegexHostAbi.exercise_source

    def test_original_complete_advanced_source(self):
        source = ROOT/'tests/test_hashmap_set_advanced.nano'
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        self.exercise_source(source.read_text())
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_every_original_advanced_assertion_in_an_active_main(self):
        original = (ROOT/'tests/test_hashmap_set_advanced.nano').read_text()
        imports, rest = original.split('fn main() -> int { return 0 }', 1)
        self.assertTrue(rest.lstrip().startswith('shadow main {'))
        body = rest.split('shadow main {', 1)[1].rstrip()
        self.assertTrue(body.endswith('}'))
        # I keep the original immutable fixture separate; this additional main
        # executes every original shadow statement through the native consumers.
        self.exercise_source(imports+'fn main()->int{'+body[:-1]+'return 0}\n'
                             'shadow main{assert (== (main) 0)}\n')

    def test_alias_retirement_and_snapshot_lifetimes(self):
        self.exercise_source('''from "std/collections/hashmap.nano" import HashMap, map_new, map_put, map_get, map_has, map_size, map_keys, map_values, map_clear, map_free
from "std/collections/set.nano" import Set, set_new, set_add, set_has, set_size, set_values, set_clear, set_free
fn main()->int{
 let initial:HashMap=(map_new)
 let alias:HashMap=initial
 let updated:HashMap=(map_put alias "key" "value")
 assert (== (map_get initial "key") "value")
 let keys:array<string> = (map_keys updated)
 let values:array<string> = (map_values updated)
 let cleared:HashMap=(map_clear initial)
 assert (== (map_size cleared) 0)
 assert (== (at keys 0) "key") assert (== (at values 0) "value")
 (map_free alias)
 assert (== (map_size initial) 0) assert (not (map_has updated "key"))
 assert (== (map_get updated "key") "")
 (map_free initial)
 assert (== (at keys 0) "key") assert (== (at values 0) "value")
 let members:Set=(set_new)
 let kept:Set=(set_add members "member")
 let snapshot:array<string> = (set_values kept)
 let emptied:Set=(set_clear members)
 assert (== (set_size emptied) 0)
 (set_free kept)
 assert (== (set_size members) 0) assert (not (set_has members "member"))
 (set_free members)
 assert (== (at snapshot 0) "member")
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_nested_records_arrays_and_function_transport(self):
        self.exercise_source('''from "std/collections/hashmap.nano" import HashMap, map_new, map_put, map_get, map_free
from "std/collections/set.nano" import Set, set_new, set_add, set_has, set_free
struct Both{mapping:HashMap,members:Set}
fn unchanged(value:Both)->Both{return value}
shadow unchanged{
 let hm:HashMap=(map_new) let st:Set=(set_new)
 let wrapped:Both=(unchanged Both{mapping:hm,members:st})
 let changed:HashMap=(map_put wrapped.mapping "shadow" "kept")
 assert (== (map_get hm "shadow") "kept")
 (map_free changed) (set_free st)
}
fn main()->int{
 let hm:HashMap=(map_new) let st:Set=(set_new)
 let wrapped:Both=Both{mapping:hm,members:st}
 let copies:array<Both> = [wrapped,(unchanged wrapped)]
 let selected:Both=(at copies 1)
 let changed:HashMap=(map_put selected.mapping "key" "kept")
 let members:Set=(set_add selected.members "member")
 assert (== (map_get hm "key") "kept") assert (set_has st "member")
 (map_free changed) (set_free members)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_same_named_handle_field_value_arms(self):
        self.exercise_source('''from "std/collections/hashmap.nano" import HashMap,map_new,map_put,map_get,map_free
fn main()->int{
 let hm:HashMap=(map_new)
 let kept:HashMap=HashMap{raw:if true {let copy:HashMap=hm copy.raw} else {hm.raw}}
 let updated:HashMap=(map_put kept "key" "same")
 assert (== (map_get hm "key") "same")
 (map_free updated)
 return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_canonical_artifact_descriptor_and_relocated_invocation(self):
        pure = routes.source_compiler_command()
        self.assertTrue(pure)
        for label, command in [('c',[ROOT/'bin/nano_virt']),('pure',pure)]:
            with self.subTest(producer=label):
                work = self.directory('collection-descriptor-'+label+'-')
                source,module = work/'main.nano',work/'main.nvm'
                source.write_text('from "std/collections/hashmap.nano" import HashMap, map_new, map_free\n'
                                  'fn main()->int{let hm:HashMap=(map_new) (map_free hm) return 0}\n'
                                  'shadow main{assert (== (main) 0)}\n')
                self.command(work,'compile',[*command,source,'--emit-nvm','-o',module],build=True)
                dump = self.command(work,'dump',[ROOT/'bin/nanoisa','dump',module]).stdout
                self.assertRegex(dump,r'(?m)^\.import "/[^\n"]+/libstd_collections\.(so|dylib)" "nl_hm_new" opaque$')
                self.assertRegex(dump,r'(?m)^\.import_kind \d+ artifact$')
                self.assertNotIn('.import "" "nl_hm_',dump)
                if label=='pure':
                    # I keep the normal tool-root discovery when the caller is
                    # outside my checkout and does not export NANOLANG_ROOT.
                    env = dict(os.environ)
                    env.pop('NANOLANG_ROOT',None)
                    from tests.native_process import run_captured
                    observed=run_captured([*command,source,'--emit-nvm','-o',work/'relocated.nvm'],
                                          cwd=work,text=True,timeout=60,build=True,env=env)
                    (work/'relocated.stdout').write_text(observed.stdout)
                    (work/'relocated.stderr').write_text(observed.stderr)
                    self.assertEqual(observed.returncode,0,observed.stdout+observed.stderr)

    def test_distinct_handle_families_preserve_prior_output(self):
        pure=routes.source_compiler_command();self.assertTrue(pure)
        bodies=[
            'let hm:HashMap=(map_new) let wrong:Set=Set{raw:hm.raw}',
            'let members:Set=(set_new) let wrong:HashMap=HashMap{raw:members.raw}',
            'let members:Set=(set_new) let wrong:HashMap=(map_put members "key" "value")',
            'let hm:HashMap=(map_new) let members:Set=(set_new) '
            'let wrong:HashMap=HashMap{raw:if true {hm.raw} else {members.raw}}',
            'let hm:HashMap=(map_new) let members:Set=(set_new) '
            'let wrong:HashMap=HashMap{raw:if false {members.raw} else {hm.raw}}'
        ]
        imports=('from "std/collections/hashmap.nano" import HashMap,map_new,map_put\n'
                 'from "std/collections/set.nano" import Set,set_new\n')
        for label,compiler in [('c',[ROOT/'bin/nano_virt']),('pure',pure)]:
            for index,body in enumerate(bodies):
                with self.subTest(producer=label,case=index):
                    work=self.directory('collection-kind-refusal-'+label+'-')
                    source,module=work/'main.nano',work/'prior.nvm'
                    source.write_text(imports+'fn main()->int{'+body+' return 0}\nshadow main{assert true}\n')
                    prior=b'I retain my prior collection kind output.\n';module.write_bytes(prior)
                    result=self.command(work,'refusal',[*compiler,source,'--emit-nvm','-o',module],
                                        build=True,accepted=False)
                    self.assertGreater(result.returncode,0,result.stdout+result.stderr)
                    self.assertEqual(module.read_bytes(),prior)

    def test_noncanonical_opaque_declarations_preserve_prior_output(self):
        pure = routes.source_compiler_command()
        self.assertTrue(pure)
        for opaque in ['NLHashMap','NLStringSet','NLStringBuilder','Arbitrary']:
            with self.subTest(opaque=opaque):
                work=self.directory('collection-impostor-')
                source,module=work/'main.nano',work/'prior.nvm'
                # My unrelated opaque extern is unused; the record transport
                # must refuse independently of native provider availability.
                source.write_text('opaque type '+opaque+'\nstruct Wrapped{raw:'+opaque+'}\n'
                                  'extern fn nl_hm_new()->'+opaque+'\n'
                                  'fn main()->int{unsafe{let raw:'+opaque+'=(nl_hm_new) '
                                  'let value:Wrapped=Wrapped{raw:raw}} return 0}\n'
                                  'shadow main{assert (== (main) 0)}\n')
                prior=b'I retain my prior collection product.\n';module.write_bytes(prior)
                observed=self.command(work,'refusal',[*pure,source,'--emit-nvm','-o',module],
                                      build=True,accepted=False)
                self.assertGreater(observed.returncode,0,observed.stdout+observed.stderr)
                self.assertEqual(module.read_bytes(),prior)
