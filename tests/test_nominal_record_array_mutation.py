"""I check the target record declaration before publishing an array mutation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SELFHOST = Path(os.environ.get('NANOLANG_SELFHOST_COMPILER', ROOT / 'bin/nanoc_stage2'))
DECL = '''struct First { a: int }
struct Second { a: int }
struct Holder { items: array<First> }
fn first() -> First { return First { a: 42 } }
shadow first { assert (== (first).a 42) }
fn second() -> Second { return Second { a: 42 } }
shadow second { assert (== (second).a 42) }
fn identity(value: T) -> T { return value }
shadow identity { assert (== (identity 7) 7) }
fn receiver(items: array<First>) -> array<First> { return items }
shadow receiver { let items: array<First> = [] assert (== (array_length (receiver items)) 0) }
'''
LOCALS = '''let items: array<First> = [First { a: 1 }, First { a: 2 }]
let alias: array<First> = items
let holder: Holder = Holder { items: items }
let nested: array<array<First>> = [items]
let wrong: Second = Second { a: 42 }
let right: First = First { a: 42 }
let callback: fn() -> Second = second
'''
END = '\nshadow main { assert (== (main) 0) }\n'


class NominalRecordArrayMutation(unittest.TestCase):
    def routes(self):
        return [('c', ROOT / 'bin/nanoc_c', []),
                ('vm', ROOT / 'bin/nano_virt', ['--emit-nvm']),
                ('selfhost', SELFHOST, ['--emit-nvm'])]

    def exercise(self, text, accepted, files=None, refusal='array mutation', diagnostic='E001'):
        with tempfile.TemporaryDirectory(prefix='nano-nominal-mutation-') as temporary:
            work = Path(temporary)
            source = work/'main.nano'
            source.write_text(text)
            for name, content in (files or {}).items():
                (work/name).write_text(content)
            for name, compiler, flags in self.routes():
                with self.subTest(route=name):
                    self.assertTrue(compiler.is_file(), str(compiler))
                    output = work/(name+'.out')
                    output.write_bytes(b'prior-output')
                    result = subprocess.run([str(compiler), str(source), *flags, '-o', str(output)],
                        cwd=ROOT, capture_output=True, text=True, timeout=180)
                    if not accepted:
                        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                        self.assertRegex(result.stdout+result.stderr, diagnostic)
                        if refusal is not None:
                            self.assertIn(refusal, result.stdout+result.stderr)
                        self.assertEqual(output.read_bytes(), b'prior-output')
                    else:
                        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                        args = [output] if name == 'c' else [ROOT/'bin/nano_vm', output]
                        run = subprocess.run(list(map(str, args)), cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
                        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

    def test_distinct_record_values_refuse_before_publication(self):
        cases = [('literal', 'items', 'Second { a: 42 }'),
                 ('local', 'alias', 'wrong'),
                 ('function', 'items', '(second)'),
                 ('generic', 'items', '(identity wrong)'),
                 ('indirect', 'items', '(callback)'),
                 ('if', 'items', '(if true { wrong } else { Second { a: 42 } })'),
                 ('cond', 'items', '(cond (true wrong) (else Second { a: 42 }))'),
                 ('field', 'holder.items', 'wrong'),
                 ('nested', '(at nested 0)', 'wrong'),
                 ('returned', '(receiver items)', 'wrong')]
        for label, target, value in cases:
            with self.subTest(case=label):
                self.exercise(DECL+'fn main() -> int {\n'+LOCALS+
                    '(array_set '+target+' 0 '+value+')\nreturn 0\n}'+END, False)
        self.exercise(DECL+'''fn replace(items: array<First>) -> void {
 (array_set items 0 Second { a: 42 })
}
shadow replace { let items: array<First> = [First { a: 1 }] (replace items) assert (== (at items 0).a 42) }
fn main() -> int { return 0 }
'''+END, False)

    def test_matching_record_values_execute_with_aliases_and_neighbors(self):
        cases = [('literal', 'items', 'First { a: 42 }'),
                 ('anonymous', 'items', '{ a: 42 }'),
                 ('local', 'alias', 'right'),
                 ('function', 'items', '(first)'),
                 ('generic', 'items', '(identity right)'),
                 ('if', 'items', '(if true { right } else { First { a: 42 } })'),
                 ('cond', 'items', '(cond (true right) (else First { a: 42 }))'),
                 ('field', 'holder.items', 'right'),
                 ('nested', '(at nested 0)', 'right'),
                 ('returned', '(receiver items)', 'right')]
        for label, target, value in cases:
            with self.subTest(case=label):
                self.exercise(DECL+'fn main() -> int {\n'+LOCALS+
                    '(array_set '+target+' 0 '+value+')\n'
                    'assert (== (at items 0).a 42) assert (== (at alias 0).a 42) '
                    'assert (== (at items 1).a 2) return 0\n}'+END, True)

    def test_control_flow_receivers_preserve_record_identity(self):
        targets = [
            '(if true { items } else { alias })',
            '(if false { alias } else { items })',
            '(if true { items } else { [] })',
            '(if false { [] } else { items })',
            '(match 1 { 1 => { items } _ => { [] } })',
            '(match 1 { 0 => { [] } _ => { items } })',
            '(cond (false alias) (true items) (else alias))',
            '(match 1 { 1 => { items } _ => { alias } })',
            '(if true { let chosen: array<First> = items chosen } else { alias })',
            '(cond (false alias) (else (if true { items } else { alias })))',
        ]
        for target in targets:
            for accepted, value in ((False, 'wrong'), (True, 'right'), (True, '{ a: 42 }'),
                                    (True, '(if true { { a: 42 } } else { { a: 43 } })')):
                with self.subTest(target=target, accepted=accepted, value=value):
                    self.exercise(DECL+'fn main() -> int {\n'+LOCALS+
                        '(array_set '+target+' 0 '+value+')\n'+
                        ('assert (== (at items 0).a 42) assert (== (at items 1).a 2) ' if accepted else '')+
                        'return 0 }'+END, accepted, refusal=None, diagnostic=r'E001|E0001')

    def test_unselected_receiver_branches_preserve_record_identity(self):
        targets = [
            '(if true { items } else { others })',
            '(if false { others } else { items })',
            '(cond (true items) (false others) (else alias))',
            '(match 1 { 1 => { items } _ => { others } })',
        ]
        for target in targets:
            with self.subTest(target=target):
                self.exercise(DECL+'fn main() -> int {\n'+LOCALS+
                    'let others: array<Second> = [wrong]\n'+
                    '(array_set '+target+' 0 right) return 0 }'+END,
                    False, refusal=None, diagnostic=r'E001|E0001')

    def test_receiver_returns_skip_index_and_value_effects(self):
        template = '''struct First { a:int }
fn position(log:array<int>)->int{(array_set log 0 (+ (at log 0) 1)) return 0}
shadow position{let log:array<int> = [0] assert (== (position log) 0) assert (== (at log 0) 1)}
fn value(log:array<int>)->First{(array_set log 0 (+ (at log 0) 10)) return First{a:42}}
shadow value{let log:array<int> = [0] assert (== (value log).a 42) assert (== (at log 0) 10)}
fn replace(flag:int,items:array<First>,log:array<int>)->int{
(array_set TARGET (position log) (value log)) return 9
}
shadow replace{
let items:array<First> = [First{a:1}] let log:array<int> = [0]
assert (== (replace 1 items log) 7) assert (== (at log 0) 0) assert (== (at items 0).a 1)
assert (== (replace 0 items log) 9) assert (== (at log 0) 11) assert (== (at items 0).a 42)
}
fn main()->int{let items:array<First> = [First{a:1}] let log:array<int> = [0] assert (== (replace 1 items log) 7) assert (== (at log 0) 0) assert (== (replace 0 items log) 9) assert (== (at log 0) 11) assert (== (at items 0).a 42) return 0}
shadow main{assert (== (main) 0)}
'''
        for target in ('(if (== flag 1) {return 7} else {items})',
                       '(cond ((== flag 1) (if true {return 7} else {return 7})) (else items))',
                       '(match flag {1=>{return 7} _=>{items}})'):
            with self.subTest(target=target):
                self.exercise(template.replace('TARGET', target), True)

    def test_discarded_statement_branches_keep_their_types(self):
        self.exercise('''fn number()->int{return 1}
shadow number{assert (== (number) 1)}
fn text()->string{return "discarded"}
shadow text{assert (== (text) "discarded")}
fn discard(flag:bool)->void{if flag {(number)} else {(text)}}
shadow discard{(discard true) (discard false) assert true}
fn main()->int{(discard true) (discard false) return 0}
shadow main{assert (== (main) 0)}
''', True)

    def test_selected_empty_receiver_values_keep_storage_type(self):
        for target in ('(if empty { [] } else { items })',
                       '(match empty { 1 => { [] } _ => { items } })'):
            boolean = target.startswith('(if')
            argument_type = 'bool' if boolean else 'int'
            yes, no = ('true', 'false') if boolean else ('1', '0')
            with self.subTest(target=target):
                self.exercise(DECL+'fn choose(empty:'+argument_type+', items:array<First>)->array<First>{return '+target+'}\n'+
                    'shadow choose{let items:array<First> = [First{a:1}] assert (== (array_length (choose '+yes+' items)) 0)}\n'+
                    'fn main()->int{let items:array<First> = [First{a:1}] '
                    'let fresh:array<First> = (array_push (choose '+yes+' items) First{a:7}) '
                    '(array_set fresh 0 First{a:42}) assert (== (at fresh 0).a 42) '
                    'assert (== (at items 0).a 1) '
                    '(array_set (choose '+no+' items) 0 First{a:9}) assert (== (at items 0).a 9) return 0}'+END, True)

    def test_record_value_context_keeps_return_and_guard_checks(self):
        for value in ('(if true { return "wrong" } else { right })',
                      '(match 1 { _ if 7 => { right } _ => { right } })'):
            with self.subTest(value=value):
                self.exercise(DECL+'fn main()->int{'+LOCALS+
                    '(array_set items 0 '+value+') return 0}'+END, False,
                    refusal=None, diagnostic=r'E001|E0001')

    def test_imported_declarations_preserve_module_identity(self):
        files = {}
        for module in ('Left', 'Right'):
            files[module.lower()+'.nano'] = ('module '+module+'\npub struct Item { a: int }\n'
                'pub fn item() -> Item { return Item { a: 42 } }\n'
                'shadow item { assert (== (item).a 42) }\n'
                'pub fn items() -> array<Item> { return [Item { a: 1 }] }\n'
                'shadow items { assert (== (array_length (items)) 1) }\n'
                'pub fn forward_items(values: array<Item>) -> array<Item> { return values }\n'
                'shadow forward_items { let values: array<Item> = [] assert (== (array_length (forward_items values)) 0) }\n')
        for owner in ('left', 'right'):
            for value in (owner+'.Item { a: 42 }', '('+owner+'.item)'):
                for target in ('items', '(left.items)', '(left.forward_items items)',
                               '(if true { items } else { (left.items) })',
                               '(cond (false (left.items)) (else items))',
                               '(match 1 { 1 => { items } _ => { (left.forward_items items) } })'):
                    with self.subTest(owner=owner, value=value, target=target):
                        self.exercise('module "left.nano" as left\nmodule "right.nano" as right\n'
                            'fn main() -> int { let items: array<left.Item> = (left.items) '
                            '(array_set '+target+' 0 '+value+') '+
                            ('assert (== (at items 0).a 42) ' if owner == 'left' and target != '(left.items)' else '')+
                            'assert (== (array_length items) 1) return 0 }'+END,
                            owner == 'left', files)
        self.exercise('module "left.nano" as left\nmodule "right.nano" as right\n'
            'fn main()->int{let items:array<left.Item> = (left.items) '
            '(array_set (if true {items} else {(right.items)}) 0 left.Item{a:42}) return 0}'+END,
            False, files, refusal=None, diagnostic=r'E001|E0001')


    def test_lexical_call_and_forbidden_redefinition_keep_their_rules(self):
        for lexical in (False, True):
            name = 'accept' if lexical else 'array_set'
            declaration = ('fn '+name+'(items: array<First>, position: int, item: Second) -> int { return item.a }\n'
                'shadow '+name+' { let items: array<First> = [] assert (== ('+name+' items 0 Second { a: 42 }) 42) }\n')
            binding = 'let array_set: fn(array<First>,int,Second)->int = accept\n' if lexical else ''
            with self.subTest(lexical=lexical):
                self.exercise(DECL+declaration+'fn main() -> int {\n'+LOCALS+binding+
                    'assert (== (array_set items 0 wrong) 42) assert (== (at items 0).a 1) return 0 }'+END, lexical,
                    refusal="redefine built-in function 'array_set'")
        self.exercise(DECL+"""union Choice { Some { value: int }, None {} }
fn accept(items: array<First>, position: int, item: Second) -> int { return item.a }
shadow accept { let items: array<First> = [] assert (== (accept items 0 Second { a: 42 }) 42) }
fn alternate(items: array<First>, position: int, item: Second) -> int { return (+ item.a 100) }
shadow alternate { let items: array<First> = [] assert (== (alternate items 0 Second { a: 42 }) 142) }
fn main() -> int {
 let items: array<First> = [First { a: 1 }]
 let mut array_set: fn(array<First>,int,Second)->int = accept
 let choice: Choice = Choice.Some { value: 0 }
 let selected: int = (array_set items (match choice {
  Some(value) => { set array_set alternate value.value }
  None(empty) => { 0 }
 }) Second { a: 42 })
 assert (== selected 42)
 assert (== (array_set items 0 Second { a: 42 }) 142)
 assert (== (at items 0).a 1)
 return 0
}
"""+END, True)


if __name__ == '__main__':
    unittest.main()
