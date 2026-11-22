"""I keep emission-time type metadata within its original lexical source scope."""
from pathlib import Path
import subprocess
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from tests import test_nominal_record_array_mutation as nominal
from tests import nanolang_translator_routes as routes

ROOT = Path(__file__).resolve().parents[1]
FLOAT = '''fn compute()->float{return 3.5}
shadow compute{assert (== (compute) 3.5)}
'''
MAIN = '''fn main()->int{assert (== (+ (compute) (compute)) 7.0) return 0}
shadow main{assert (== (main) 0)}
'''


class EmissionScopes(unittest.TestCase):
    routes = nominal.NominalRecordArrayMutation.routes

    def exercise(self, source, accepted=True, files=None, legacy_native=True):
        if not accepted:
            nominal.NominalRecordArrayMutation.exercise(self, source, False, files,
                refusal=None, diagnostic=r'E001|E0001|E0010')
            return
        work = Path(tempfile.mkdtemp(prefix='nano-emission-scope-'))
        fixture = work/'main.nano'; fixture.write_text(source)
        for name, data in (files or {}).items(): (work/name).write_text(data)
        original = routes.run_captured
        commands = []
        def checked(command, **kwargs):
            result = original(command, **kwargs)
            index = len(commands)
            commands.append(dict(args=list(map(str,command)), exit=result.returncode,
                                 build=kwargs.get('build',False), timeout=kwargs.get('timeout')))
            for name in ('stdout','stderr'): (work/(str(index)+'.'+name)).write_bytes(getattr(result,name))
            (work/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            if not result.returncode:
                self.assertNotIn(b'I require a function value for a bound call.', result.stdout+result.stderr)
                self.assertNotIn(b'Field access requires a struct value.', result.stdout+result.stderr)
            return result
        environment = {}
        if not (os.environ.get('NANOLANG_TRANSLATOR_SOURCE_COMPILER') or os.environ.get('NANOLANG_TRANSLATOR_SOURCE_MODULE')):
            environment['NANOLANG_TRANSLATOR_SOURCE_COMPILER'] = str(nominal.SELFHOST)
        with patch.dict(os.environ,environment), patch.object(routes,'run_captured',checked):
            products = routes.build_routes(fixture,work,legacy_native=legacy_native)
            self.assertEqual(len(products),5 if legacy_native else 4)
            for label, command in products:
                with self.subTest(route=label): routes.checked(command,timeout=30)

    def test_prior_parameter_keeps_float_return_and_opcode(self):
        source = ('fn forward(compute:bool)->bool{return compute}\n'
                  'shadow forward{assert (forward true)}\n'+FLOAT+MAIN)
        for text in (source, source.replace('\n', ' ')):
            with self.subTest(single_line='\n' not in text): self.exercise(text)

    def test_prior_record_parameter_keeps_declared_call_diagnostics_clean(self):
        self.exercise('''struct Item{n:int}
fn forward(union_transfer:bool)->bool{return union_transfer}
shadow forward{assert (forward true)}
fn union_transfer()->Item{return Item{n:42}}
shadow union_transfer{assert (== (union_transfer).n 42)}
fn main()->int{assert (== (union_transfer).n 42) return 0}
shadow main{assert (== (main) 0)}
''')

    def test_prior_function_local_keeps_float_return_and_opcode(self):
        self.exercise('fn forward()->bool{let compute:bool=true return compute}\n'
                      'shadow forward{assert (forward)}\n'+FLOAT+MAIN)

    def test_nested_parameter_local_and_lambda_do_not_escape(self):
        for binding in (
            'fn nested(compute:bool)->bool{return compute} assert (nested true)',
            'fn nested()->bool{let compute:bool=true return compute} assert (nested)',
            'let nested:fn(bool)->bool = fn(compute:bool)->bool{return compute} assert (nested true)',
        ):
            with self.subTest(binding=binding):
                self.exercise(FLOAT+'fn main()->int{'+binding+
                    ' assert (== (+ (compute) (compute)) 7.0) return 0}\n'
                    'shadow main{assert (== (main) 0)}\n', legacy_native=False)

    def test_closed_block_local_does_not_hide_declared_call(self):
        self.exercise(FLOAT+'''fn main()->int{
 if true {let compute:bool=true assert compute}
 let number:int=3 assert (== (number + 2) 5)
 assert (== (+ (compute) (compute)) 7.0) return 0
}
shadow main{assert (== (main) 0)}
''')

    def test_imported_parameter_and_declared_call_keep_module_ownership(self):
        helper = ('pub fn forward(compute:bool)->bool{return compute}\n'
            'shadow forward{assert (forward true)}\n'
            'pub '+FLOAT)
        self.exercise('''module "scope_helper.nano" as helper
fn main()->int{let compute:bool=false assert (not compute)
 assert (helper.forward true)
 assert (== (+ (helper.compute) (helper.compute)) 7.0) return 0}
shadow main{assert (== (main) 0)}
''', files={'scope_helper.nano':helper})

    def test_actual_lexical_callbacks_keep_precedence_and_captures(self):
        self.exercise(FLOAT+'''fn main()->int{
 let base:float=2.5
 let compute:fn()->float = fn()->float{return base}
 assert (== (+ (compute) (compute)) 5.0) return 0}
shadow main{assert (== (main) 0)}
''', legacy_native=False)

    def test_actual_noncallable_bindings_still_refuse_before_publication(self):
        for source in (
            'fn forward(compute:bool)->bool{return (compute)}\nshadow forward{assert true}\n',
            'fn forward()->bool{let compute:bool=true return (compute)}\nshadow forward{assert true}\n',
        ):
            with self.subTest(source=source):
                self.exercise(source+FLOAT+'fn main()->int{return 0}\nshadow main{assert true}\n', False)


    def test_nonfunction_storage_kinds_refuse_at_zero_and_nonzero_arity(self):
        for declared, value in (('int','1'),('bool','true'),('string','"text"'),
                                ('Item','Item{n:1}'),('array<int>','[1]')):
            for arguments in ('',' 1'):
                for dead_branch in (False,True):
                    call = '(target'+arguments+')'
                    if dead_branch: call = 'if false {'+call+'}'
                    source = ('struct Item{n:int}\nfn invalid()->void{let target:'+declared+' = '+value+
                        ' '+call+'}\nshadow invalid{assert true}\nfn main()->int{return 0}\nshadow main{assert true}\n')
                    with self.subTest(type=declared, arguments=arguments, dead=dead_branch):
                        self.exercise(source,False)


if __name__ == '__main__':
    unittest.main()
