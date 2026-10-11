"""I retain aggregate union payloads through calls, matching and shared mutation."""
import unittest
from tests import test_cseed_generic_functions as base

SHADOW = '\nshadow main { assert (== (main) 0) }\n'


class UnionAggregatePayloads(unittest.TestCase):
    run_command = base.CseedGenericFunctions.run_command
    execute = base.CseedGenericFunctions.execute

    def test_record_and_generic_record_results(self):
        for union, annotation, constructor in [
            ('union Box { Some { value:Item } }', 'Box', 'Box'),
            ('union Box<T> { Some { value:T } }', 'Box<Item>', 'Box<Item>'),
        ]:
            with self.subTest(annotation=annotation):
                self.execute('struct Item { label:string, values:array<int> }\n'+union+f'''
fn make()->{annotation} {{ return {constructor}.Some {{value:Item {{label:"kept", values:[7,9]}}}} }}
shadow make {{ assert true }}
fn read(box:{annotation})->int {{ match box {{ Some(p)=>{{ assert (== p.value.label "kept") return (at p.value.values 1) }} }} }}
shadow read {{ assert (== (read (make)) 9) }}
fn main()->int {{ let box:{annotation} = (make) assert (== (read box) 9) return 0 }}
'''+SHADOW)

    def test_primitive_lists_keep_shared_mutation(self):
        for element, value in [('int','7'), ('string','"kept"'), ('bool','true')]:
            with self.subTest(element=element):
                self.execute(f'''union Box<T> {{ Some {{value:T}} }}
fn grow(box:Box<List<{element}>>)->int {{ match box {{ Some(p)=>{{ (list_{element}_push p.value {value}) return (list_{element}_length p.value) }} }} }}
shadow grow {{ let values:List<{element}>  = (list_{element}_new) assert (== (grow Box<List<{element}>>.Some {{value:values}}) 1) }}
fn main()->int {{ let values:List<{element}>  = (list_{element}_new) let box:Box<List<{element}>> =Box<List<{element}>>.Some {{value:values}} assert (== (grow box) 1) assert (== (list_{element}_length values) 1) return 0 }}
'''+SHADOW)

    def test_record_arrays_escape_and_mutate(self):
        self.execute('''struct Item { value:int }
union Box<T> { Some {value:T} }
fn make()->Box<array<Item>> { return Box<array<Item>>.Some {value:[Item {value:7}]} }
shadow make { assert true }
fn read(box:Box<array<Item>>)->int { match box { Some(p)=>{ (array_set p.value 0 Item {value:9}) return (at p.value 0).value } } }
shadow read { assert (== (read (make)) 9) }
fn main()->int { let box:Box<array<Item>>  = (make) assert (== (read box) 9) return 0 }
'''+SHADOW)

    def test_tuple_payload(self):
        self.execute('''union Box<T> { Some {value:T} }
fn make()->Box<(int,string)> { return Box<(int,string)>.Some {value:(7,"kept")} }
shadow make { assert true }
fn read(box:Box<(int,string)>)->int { match box { Some(p)=>{ assert (== p.value.1 "kept") return p.value.0 } } }
shadow read { assert (== (read (make)) 7) }
fn main()->int { assert (== (read (make)) 7) return 0 }
'''+SHADOW)

    def test_alternating_record_and_union_results(self):
        self.execute('''struct Leaf {label:string}
union Inner { Some {leaf:Leaf} }
struct Middle {inner:Inner}
union Outer { Some {middle:Middle} }
fn make()->Outer { return Outer.Some {middle:Middle {inner:Inner.Some {leaf:Leaf {label:"kept"}}}} }
shadow make { assert true }
fn read(outer:Outer)->string { match outer { Some(p)=>{ match p.middle.inner { Some(q)=>{ return q.leaf.label } } } } }
shadow read { assert (== (read (make)) "kept") }
fn main()->int { assert (== (read (make)) "kept") return 0 }
'''+SHADOW)

    def test_imported_equal_record_names_keep_payload_identity(self):
        self.execute('''module "a.nano" as a
module "b.nano" as b
union Box<T> { Some {value:T} }
fn read(box:Box<a.Item>)->string { match box { Some(p)=>{ return p.value.label } } }
shadow read { assert (== (read Box<a.Item>.Some {value:(a.make)}) "left") }
fn main()->int { let box:Box<a.Item> = Box<a.Item>.Some {value:(a.make)} assert (== (read box) "left") return 0 }
'''+SHADOW, files={
            'a.nano': 'pub struct Item {label:string}\npub fn make()->Item {return Item {label:"left"}}\nshadow make {assert true}\n',
            'b.nano': 'pub struct Item {label:string}\npub fn make()->Item {return Item {label:"right"}}\nshadow make {assert true}\n',
        })

    def test_wrong_nominal_constructor_arguments_preserve_output(self):
        import tempfile
        import subprocess
        from pathlib import Path
        for annotation, construction in [
            ('Box<Left>', 'Box<Right>.Some {value:Right {value:7}}'),
            ('Box<array<Left>>', 'Box<array<Right>>.Some {value:[Right {value:7}]}'),
        ]:
            with self.subTest(annotation=annotation), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp)/'main.nano', Path(tmp)/'output'
                source.write_text('struct Left {value:int}\nstruct Right {value:int}\nunion Box<T> {Some {value:T}}\n'
                    f'fn main()->int {{ let value:{annotation} = {construction} return 0 }}'+SHADOW)
                output.write_bytes(b'prior artifact')
                result = subprocess.run([str(base.COMPILER), str(source), *base.MODE, '-o', str(output)],
                                        cwd=base.ROOT, capture_output=True, text=True, timeout=120)
                self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertEqual(output.read_bytes(), b'prior artifact')
                self.assertNotIn('C compilation failed', result.stdout+result.stderr)

    def test_array_payload_element_kinds(self):
        for element, literal, expected in [
            ('bool','[true,false]','false'),
            ('float','[1.25,7.5]','7.5'),
            ('string','["first","kept"]','"kept"'),
        ]:
            with self.subTest(element=element):
                self.execute(f'''union Box<T> {{Some {{value:T}}}}
fn read(box:Box<array<{element}>>)->{element} {{ match box {{Some(p)=>{{return (at p.value 1)}}}} }}
shadow read {{assert (== (read Box<array<{element}>>.Some {{value:{literal}}}) {expected})}}
fn main()->int {{let box:Box<array<{element}>> = Box<array<{element}>>.Some {{value:{literal}}} assert (== (read box) {expected}) return 0}}
'''+SHADOW)

    def test_map_payload_retains_shared_storage(self):
        for element, value in [('int','7'),('string','"kept"')]:
            with self.subTest(element=element):
                self.execute(f'''union Box<T> {{Some {{value:T}}}}
fn update(box:Box<HashMap<string,{element}>>)->int {{match box {{Some(p)=>{{(map_put p.value "key" {value}) return 0}}}}}}
shadow update {{let values:HashMap<string,{element}> = (map_new) assert (== (update Box<HashMap<string,{element}>>.Some {{value:values}}) 0)}}
fn main()->int {{let values:HashMap<string,{element}> = (map_new) let box:Box<HashMap<string,{element}>> = Box<HashMap<string,{element}>>.Some {{value:values}} assert (== (update box) 0) assert (== (map_get values "key") {value}) return 0}}
'''+SHADOW)

    def test_record_list_payload_mutates_shared_storage(self):
        self.execute('''struct Item {label:string}
union Box<T> {Some {value:T}}
fn update(box:Box<List<Item>>)->string { match box {Some(p)=>{(list_Item_push p.value Item {label:"kept"}) return (list_Item_get p.value 0).label}} }
shadow update {let values:List<Item> = (list_Item_new) assert (== (update Box<List<Item>>.Some {value:values}) "kept")}
fn main()->int {let values:List<Item> = (list_Item_new) let box:Box<List<Item>> = Box<List<Item>>.Some {value:values} assert (== (update box) "kept") assert (== (list_Item_length values) 1) return 0}
'''+SHADOW)
