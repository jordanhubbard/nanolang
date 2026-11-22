"""I retain runtime descriptors, explicit providers and ordinary regex names."""
import json
import unittest
from pathlib import Path

from tests import nanolang_translator_routes as routes
from tests import test_native_regex_host_abi as abi

CONTRACTS = abi.CONTRACTS
ROOT = abi.ROOT


class RegexSourceProviders(unittest.TestCase):
    command = abi.NativeRegexHostAbi.command
    directory = abi.NativeRegexHostAbi.directory
    exercise_source = abi.NativeRegexHostAbi.exercise_source

    def producers(self):
        pure = routes.source_compiler_command()
        self.assertTrue(pure, 'I require an explicitly selected pure compiler product.')
        return [('c', [ROOT/'bin/nano_virt']), ('pure', pure)]

    def test_all_nine_runtime_descriptors(self):
        for label, command in self.producers():
            with self.subTest(producer=label):
                work = self.directory('descriptors-'+label+'-')
                source, module = work/'main.nano', work/'main.nvm'
                source.write_text('from "std/regex/regex.nano" import Regex, compile, matches, find, find_all, groups, replace, replace_all, split, free\n'
                                  'fn main()->int{let re:Regex=(compile "a")\n'
                                  'assert (matches re "a") assert (== (find re "a") 0)\n'
                                  'let positions:array<int> = (find_all re "a")\n'
                                  'let captures:array<string> = (groups re "a")\n'
                                  'let first:string=(replace re "a" "b")\n'
                                  'let all:string=(replace_all re "a" "b")\n'
                                  'let pieces:array<string> = (split re "a")\n'
                                  '(free re) return 0}\nshadow main{assert (== (main) 0)}\n')
                self.command(work, 'compile', [*command, source, '--emit-nvm', '-o', module], build=True)
                dump = self.command(work, 'dump', [ROOT/'bin/nanoisa', 'dump', module]).stdout
                imports = [line for line in dump.splitlines() if line.startswith('.import ')]
                actual = [line for line in imports if '"nl_regex_' in line]
                expected = ['.import "" '+json.dumps(name)+' '+result+
                            (' '+' '.join(parameters) if parameters else '')
                            for name, result, parameters in CONTRACTS]
                self.assertEqual(actual, expected)
                self.assertNotRegex(dump, r'(?m)^\.import_kind \d+ artifact$')

    def test_explicit_artifact_and_library_never_fall_back_to_runtime(self):
        for choice in ['artifact', 'missing-source', 'library-only']:
            for label, command in self.producers():
                with self.subTest(provider=choice, producer=label):
                    work = self.directory('provider-'+choice+'-'+label+'-')
                    provider = work/'provider'
                    provider.mkdir()
                    api = provider/'api.nano'
                    api.write_text('opaque type Regex\npub extern fn nl_regex_compile(pattern:string)->Regex\n')
                    metadata = {'name':'regex-selected-provider'}
                    if choice == 'library-only':
                        metadata['system_libs'] = ['m']
                    else:
                        metadata['c_sources'] = ['api.c' if choice == 'artifact' else 'missing.c']
                    (provider/'module.json').write_text(json.dumps(metadata))
                    (provider/'api.c').write_text('void *nl_regex_compile(const char *pattern) { return (void *)pattern; }\n')
                    source, module, generated = work/'main.nano', work/'main.nvm', work/'prior.c'
                    source.write_text('from '+json.dumps(str(api))+' import Regex, nl_regex_compile\n'
                                      'fn main()->int{unsafe{let compiled:Regex=(nl_regex_compile "a")} return 0}\n'
                                      'shadow main{assert (== (main) 0)}\n')
                    prior_module = b'I retain my prior source product.\n'
                    module.write_bytes(prior_module)
                    result = self.command(work, 'compile', [*command, source, '--emit-nvm', '-o', module],
                                          build=True, accepted=False)
                    if choice == 'missing-source' or (choice == 'library-only' and label == 'pure'):
                        self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                        self.assertEqual(module.read_bytes(), prior_module)
                        continue
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    dump = self.command(work, 'dump', [ROOT/'bin/nanoisa', 'dump', module]).stdout
                    self.assertNotIn('.import "" "nl_regex_compile"', dump)
                    self.assertRegex(dump, r'(?m)^\.import "[^"]+" "nl_regex_compile" opaque string$')
                    if choice == 'artifact':
                        self.assertRegex(dump, r'(?m)^\.import_kind \d+ artifact$')
                    prior = b'I retain my prior native translation.\n'
                    generated.write_bytes(prior)
                    refused = self.command(work, 'native-refusal', [ROOT/'bin/nvm2c', module, '-o', generated], accepted=False)
                    self.assertGreater(refused.returncode, 0, refused.stdout+refused.stderr)
                    self.assertEqual(generated.read_bytes(), prior)

    def test_wrong_array_family_does_not_select_the_runtime(self):
        for label, command in self.producers():
            with self.subTest(producer=label):
                work = self.directory('wrong-family-'+label+'-')
                api = work/'api.nano'
                api.write_text('opaque type Regex\npub extern fn nl_regex_compile(pattern:string)->Regex\n'
                               'pub extern fn nl_regex_find_all(re:Regex,text:string)->array<string>\n')
                source, module = work/'main.nano', work/'main.nvm'
                source.write_text('from '+json.dumps(str(api))+' import Regex, nl_regex_compile, nl_regex_find_all\n'
                                  'fn main()->int{unsafe{let re:Regex=(nl_regex_compile "a") '
                                  'let positions:array<string> = (nl_regex_find_all re "a")} return 0}\n'
                                  'shadow main{assert (== (main) 0)}\n')
                prior = b'I retain my prior array-family product.\n'
                module.write_bytes(prior)
                result = self.command(work, 'compile', [*command, source, '--emit-nvm', '-o', module],
                                      build=True, accepted=False)
                if label == 'pure':
                    self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                    self.assertEqual(module.read_bytes(), prior)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    dump = self.command(work, 'dump', [ROOT/'bin/nanoisa', 'dump', module]).stdout
                    self.assertNotIn('.import "" "nl_regex_find_all"', dump)
                    generated = work/'prior.c'
                    generated.write_bytes(prior)
                    refused = self.command(work, 'native-refusal', [ROOT/'bin/nvm2c', module, '-o', generated], accepted=False)
                    self.assertGreater(refused.returncode, 0, refused.stdout+refused.stderr)
                    self.assertEqual(generated.read_bytes(), prior)

    def test_same_spelling_struct_is_not_the_opaque_runtime_type(self):
        for label, command in self.producers():
            with self.subTest(producer=label):
                work = self.directory('struct-not-opaque-'+label+'-')
                api = work/'api.nano'
                api.write_text('struct Regex{code:int}\npub extern fn nl_regex_compile(pattern:string)->Regex\n')
                source, module, generated = work/'main.nano', work/'prior.nvm', work/'prior.c'
                source.write_text('from '+json.dumps(str(api))+' import Regex, nl_regex_compile\n'
                                  'fn main()->int{unsafe{let compiled:Regex=(nl_regex_compile "a")} return 0}\n'
                                  'shadow main{assert (== (main) 0)}\n')
                prior = b'I retain my prior nominal-type product.\n'
                module.write_bytes(prior)
                result = self.command(work, 'compile', [*command, source, '--emit-nvm', '-o', module],
                                      build=True, accepted=False)
                if label == 'pure':
                    self.assertGreater(result.returncode, 0, result.stdout+result.stderr)
                    self.assertIn('Regex opaque type', result.stdout+result.stderr)
                    self.assertEqual(module.read_bytes(), prior)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                    dump = self.command(work, 'dump', [ROOT/'bin/nanoisa', 'dump', module]).stdout
                    self.assertNotIn('.import "" "nl_regex_compile"', dump)
                    generated.write_bytes(prior)
                    refused = self.command(work, 'native-refusal', [ROOT/'bin/nvm2c', module, '-o', generated], accepted=False)
                    self.assertGreater(refused.returncode, 0, refused.stdout+refused.stderr)
                    self.assertEqual(generated.read_bytes(), prior)

    def test_ordinary_names_direct_values_and_callbacks(self):
        names = [name.removeprefix('nl_') for name, _, _ in CONTRACTS]
        declarations = ''.join('fn '+name+'()->int{return '+str(index+11)+'}\n'
                               'shadow '+name+'{assert (== ('+name+') '+str(index+11)+')}\n'
                               for index, name in enumerate(names))
        body = ''.join('assert (== ('+name+') '+str(index+11)+')\n'
                       for index, name in enumerate(names))
        self.exercise_source('from "std/regex/regex.nano" import Regex, compile, matches, free\n'+
                             declarations+'fn apply(callback:fn()->int)->int{return (callback)}\n'
                             'shadow apply{assert (== (apply regex_compile) 11)}\n'
                             'fn main()->int{\n'+body+
                             'let chosen:fn()->int=regex_compile\nassert (== (chosen) 11)\n'
                             'assert (== (apply regex_compile) 11)\n'
                             'let re:Regex=(compile "a")\nassert (matches re "a")\n(free re)\nreturn 0}\n'
                             'shadow main{assert (== (main) 0)}\n')
