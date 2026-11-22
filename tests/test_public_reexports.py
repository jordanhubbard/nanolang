"""I retain public callable declaration identity through convenience modules."""
import json
import unittest
from tests import nanolang_translator_routes as routes
from tests import test_native_regex_host_abi as abi

ROOT = abi.ROOT
LEAF = '''module leaf
pub fn map_put(n:int)->int{return (+ n 1)}
shadow map_put{assert (== (map_put 40) 41)}
pub fn answer(n:int)->int{return (+ n 1)}
shadow answer{assert (== (answer 40) 41)}
fn hidden(n:int)->int{return n}
shadow hidden{assert (== (hidden 9) 9)}
'''


class PublicReexports(unittest.TestCase):
    command = abi.NativeRegexHostAbi.command
    directory = abi.NativeRegexHostAbi.directory
    exercise_source = abi.NativeRegexHostAbi.exercise_source

    def fixture(self, wrapper, imports, call='(map_put 40)', expected=41, *, extra=None):
        work = self.directory('public-reexport-inputs-')
        (work/'leaf.nano').write_text(LEAF)
        (work/'barrel.nano').write_text('module barrel\n'+wrapper+'\n')
        if extra:
            for name, text in extra.items():
                (work/name).write_text(text)
        source = imports.replace('@BARREL@', json.dumps(str(work/'barrel.nano')))
        source += '\nfn main()->int{assert (== '+call+' '+str(expected)+') return 0}\n'
        source += 'shadow main{assert (== (main) 0)}\n'
        return work, source

    def test_public_functions_and_selective_aliases_all_routes(self):
        for wrapper, imports, call in [
            ('import "leaf.nano"', 'from @BARREL@ import map_put', '(map_put 40)'),
            ('import "leaf.nano"', 'from @BARREL@ import answer', '(answer 40)'),
            ('from "leaf.nano" import map_put as chosen', 'from @BARREL@ import chosen', '(chosen 40)')]:
            with self.subTest(wrapper=wrapper, imports=imports):
                _, source = self.fixture(wrapper, imports, call)
                self.exercise_source(source)

    def test_multihop_public_alias_and_qualified_consumer_all_routes(self):
        work, source = self.fixture('from "middle.nano" import selected as chosen',
            'module @BARREL@ as imported', '(imported.chosen 40)',
            extra={'middle.nano': 'module middle\nfrom "leaf.nano" import answer as selected\n'})
        self.exercise_source(source)

    def test_same_spelling_retains_the_imported_declaration_all_routes(self):
        work, source = self.fixture('import "leaf.nano"', 'from @BARREL@ import answer', '(answer 40)')
        (work/'other.nano').write_text('module other\npub fn answer(n:int)->string{return "other"}\n'
                                      'shadow answer{assert (== (answer 40) "other")}\n')
        source = ('module '+json.dumps(str(work/'other.nano'))+' as other\n'+source)
        source = source.replace('fn main()->int{', 'fn main()->int{assert (== (other.answer 40) "other") ', 1)
        self.exercise_source(source)

    def test_imported_alias_callable_transport_all_routes(self):
        work, source = self.fixture('from "leaf.nano" import answer as chosen',
            'from @BARREL@ import chosen', '(callbacks.invoke chosen)',
            extra={'callbacks.nano': 'module callbacks\n'
                'pub fn invoke(action:fn(int)->int)->int{return (action 40)}\n'
                'pub fn chosen(n:int)->int{return (+ n 2)}\n'
                'shadow chosen{assert (== (chosen 40) 42)}\n'
                'fn local_answer(n:int)->int{return (+ n 1)}\n'
                'shadow local_answer{assert (== (local_answer 40) 41)}\n'
                'shadow invoke{assert (== (invoke local_answer) 41)}\n'})
        source = 'module '+json.dumps(str(work/'callbacks.nano'))+' as callbacks\n'+source
        self.exercise_source(source)

    def test_transitive_graph_growth_and_duplicate_paths_all_routes(self):
        work = self.directory('public-reexport-graph-')
        imports = []
        names = []
        for index in range(24):
            name = 'value_'+str(index)
            names.append(name)
            (work/('leaf_'+str(index)+'.nano')).write_text(
                'module leaf_'+str(index)+'\n'
                'pub fn '+name+'()->int{return '+str(index)+'}\n'
                'shadow '+name+'{assert (== ('+name+') '+str(index)+')}\n')
            imports.append('import "leaf_'+str(index)+'.nano"')
        imports.append('import "./leaf_0.nano"')
        (work/'barrel.nano').write_text('module barrel\n'+'\n'.join(imports)+'\n')
        source = 'from '+json.dumps(str(work/'barrel.nano'))+' import '+','.join(names)+'\n'
        source += 'fn main()->int{\n'
        for index, name in enumerate(names):
            source += 'assert (== ('+name+') '+str(index)+')\n'
        source += 'return 0}\nshadow main{assert (== (main) 0)}\n'
        (work/'root.nano').write_text(source)
        self.exercise_source(source)

    def test_declared_argument_contracts_preserve_prior_output(self):
        pure = routes.source_compiler_command()
        self.assertTrue(pure)
        for call in ['(map_put "wrong")', '(map_put)', '(map_put 40 2)']:
            for label, compiler in [('c', [ROOT/'bin/nano_virt']), ('pure', pure)]:
                with self.subTest(call=call, producer=label):
                    work, source = self.fixture('import "leaf.nano"', 'from @BARREL@ import map_put', call)
                    path, module = work/'main.nano', work/'prior.nvm'
                    path.write_text(source)
                    prior = b'I retain my prior callable re-export output.\n'
                    module.write_bytes(prior)
                    result = self.command(work, 'refusal-'+label, [*compiler, path, '--emit-nvm', '-o', module], build=True, accepted=False)
                    self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                    self.assertRegex(result.stdout+result.stderr, r'(?i)(argument|parameter|expects|type)')
                    self.assertEqual(module.read_bytes(), prior)

    def test_pure_private_and_scoped_names_stay_closed(self):
        pure = routes.source_compiler_command()
        self.assertTrue(pure)
        for wrapper, imports, call, value in [
            ('import "leaf.nano"', 'from @BARREL@ import hidden', '(hidden 9)', 9),
            ('module "leaf.nano" as scoped', 'from @BARREL@ import answer', '(answer 40)', 41)]:
            with self.subTest(wrapper=wrapper, imports=imports):
                work, source = self.fixture(wrapper, imports, call, value)
                path, module = work/'main.nano', work/'prior.nvm'
                path.write_text(source)
                prior = b'I retain my prior closed callable output.\n'
                module.write_bytes(prior)
                result = self.command(work, 'refusal', [*pure, path, '--emit-nvm', '-o', module], build=True, accepted=False)
                self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn('undefined function', result.stdout+result.stderr)
                self.assertEqual(module.read_bytes(), prior)

    def test_pure_local_definition_keeps_precedence(self):
        pure = routes.source_compiler_command()
        self.assertTrue(pure)
        work, source = self.fixture('import "leaf.nano"\npub fn answer(n:int)->int{return (+ n 2)}\n'
            'shadow answer{assert (== (answer 40) 42)}', 'from @BARREL@ import answer', '(answer 40)', 42)
        path, module = work/'main.nano', work/'main.nvm'
        path.write_text(source)
        self.command(work, 'compile', [*pure, path, '--emit-nvm', '-o', module], build=True)
        result = self.command(work, 'vm', [ROOT/'bin/nano_vm', module])
        self.assertEqual((result.stdout, result.stderr), ('', ''))


if __name__ == '__main__':
    unittest.main()
