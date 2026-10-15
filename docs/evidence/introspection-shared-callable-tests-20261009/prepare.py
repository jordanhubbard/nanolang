from pathlib import Path
root=Path('/private/tmp/nanolang-match-guards-20261009');out=Path('/private/tmp/nanolang-introspection-permanent-20261009')
fixture=Path('/private/tmp/nanolang-introspection-callables-20261009/all_values.nano').read_text().replace('"/private/tmp/nanolang-introspection-callables-20261009/probe.nano"','@MODULE_PATH@')
fixture=fixture.replace('shadow index { assert true }','shadow index { let before: int = evaluations assert (== (index) 0) assert (== evaluations (+ before 1)) set evaluations before }').replace('shadow returned { assert true }','shadow returned { let count: fn() -> int = (returned) assert (== (count) 1) }').replace('shadow invoke { assert true }','shadow invoke { let count: fn() -> int = (returned) assert (== (invoke count) 1) }')
(out/'module_introspection_callables.nano.txt').write_text(fixture)
s=(root/'tests/test_nanoisa_introspection.py').read_text();s=s.replace('class NanoisaIntrospection(unittest.TestCase):','CALLABLE_FIXTURE = ROOT / "tests/fixtures/module_introspection_callables.nano.txt"\n\n\nclass NanoisaIntrospection(unittest.TestCase):',1)
a=s.index('            module, c_file, binary =');b=s.index('\n    def test_all_operations',a)
block=s[a:b];s=s[:a]+'            self.exercise_product(source, work)\n\n    def exercise_product(self, source, work):\n'+''.join(line[4:]+'\n' for line in block.splitlines())+s[b:]
marker='    def test_repository_metadata_programs(self):'
new='''    def test_function_values_and_declared_module_path(self):
        with tempfile.TemporaryDirectory(prefix='nano-module-callables-') as directory:
            work = Path(directory).resolve()
            dependency = work / 'different_filename.nano'
            dependency.write_text('module callable_probe\\n'
                'pub struct Visible { value: int }\\n'
                'pub fn answer() -> int { return 7 }\\n'
                'shadow answer { assert (== (answer) 7) }\\n')
            source = work / 'main.nano'
            source.write_text(CALLABLE_FIXTURE.read_text().replace('@MODULE_PATH@', json.dumps(str(dependency))))
            self.exercise_product(source, work)

''';assert marker in s;s=s.replace(marker,new+marker,1);(out/'test_nanoisa_introspection.py').write_text(s)
