"""I order concrete tuple payloads before ordinary and generic native unions."""
from tests import test_union_aggregate_payloads as shared
import unittest


class NativeUnionTuples(unittest.TestCase):
    run_command = shared.UnionAggregatePayloads.run_command
    execute = shared.UnionAggregatePayloads.execute

    def test_ordinary_tuple_payload(self):
        self.execute('''union Pair {Some {value:(int,string)}}
fn read(value:Pair)->int {match value {Some(p)=>{assert (== p.value.1 "kept") return p.value.0}}}
shadow read {assert (== (read Pair.Some {value:(7,"kept")}) 7)}
fn main()->int {assert (== (read Pair.Some {value:(7,"kept")}) 7) return 0}
'''+shared.SHADOW)

    def test_nested_generic_tuple_payload_dependencies(self):
        self.execute('''struct Leaf {value:int}
union Box<T> {Some {value:T}}
fn read(box:Box<(Leaf,(int,string))>)->int {match box {Some(p)=>{let inner:(int,string) = p.value.1 let leaf:Leaf = p.value.0 assert (== inner.1 "kept") return (+ leaf.value inner.0)}}}
shadow read {assert (== (read Box<(Leaf,(int,string))>.Some {value:(Leaf {value:7},(9,"kept"))}) 16)}
fn main()->int {assert (== (read Box<(Leaf,(int,string))>.Some {value:(Leaf {value:7},(9,"kept"))}) 16) return 0}
'''+shared.SHADOW)

    def test_distinct_tuple_arguments_keep_distinct_layouts(self):
        self.execute('union Box<T> {Some {value:T}}\nfn first(box:Box<(int,string)>)->int {match box {Some(p)=>{assert (== p.value.1 "left") return p.value.0}}}\nshadow first {assert (== (first Box<(int,string)>.Some {value:(7,"left")}) 7)}\nfn second(box:Box<(string,int)>)->int {match box {Some(p)=>{assert (== p.value.0 "right") return p.value.1}}}\nshadow second {assert (== (second Box<(string,int)>.Some {value:("right",9)}) 9)}\nfn main()->int {assert (== (first Box<(int,string)>.Some {value:(7,"left")}) 7) assert (== (second Box<(string,int)>.Some {value:("right",9)}) 9) return 0}\nshadow main {assert (== (main) 0)}\n')

    def test_distinct_callable_arguments_keep_distinct_signatures(self):
        self.execute('''union Box<T> {Some {value:T}}
fn number(value:int)->int {return (+ value 1)}
shadow number {assert (== (number 7) 8)}
fn text(value:string)->string {return (+ value "!")}
shadow text {assert (== (text "kept") "kept!")}
fn first(box:Box<fn(int)->int>)->int {match box {Some(p)=>{let callback:fn(int)->int = p.value return (callback 7)}}}
shadow first {assert (== (first Box<fn(int)->int>.Some {value:number}) 8)}
fn second(box:Box<fn(string)->string>)->string {match box {Some(p)=>{let callback:fn(string)->string = p.value return (callback "kept")}}}
shadow second {assert (== (second Box<fn(string)->string>.Some {value:text}) "kept!")}
fn main()->int {assert (== (first Box<fn(int)->int>.Some {value:number}) 8) assert (== (second Box<fn(string)->string>.Some {value:text}) "kept!") return 0}
'''+shared.SHADOW)

    def test_callable_signatures_inside_tuple_keys(self):
        self.execute('''union Box<T> {Some {value:T}}
fn number(value:int)->int {return (+ value 1)}
shadow number {assert (== (number 7) 8)}
fn text(value:string)->string {return (+ value "!")}
shadow text {assert (== (text "kept") "kept!")}
fn first(box:Box<(fn(int)->int,int)>)->int {match box {Some(p)=>{let callback:fn(int)->int = p.value.0 return (callback p.value.1)}}}
shadow first {assert (== (first Box<(fn(int)->int,int)>.Some {value:(number,7)}) 8)}
fn second(box:Box<(fn(string)->string,string)>)->string {match box {Some(p)=>{let callback:fn(string)->string = p.value.0 return (callback p.value.1)}}}
shadow second {assert (== (second Box<(fn(string)->string,string)>.Some {value:(text,"kept")}) "kept!")}
fn main()->int {assert (== (first Box<(fn(int)->int,int)>.Some {value:(number,7)}) 8) assert (== (second Box<(fn(string)->string,string)>.Some {value:(text,"kept")}) "kept!") return 0}
'''+shared.SHADOW)

    def test_ordinary_record_list_payload(self):
        self.execute('''struct Item {value:int}
union Choice {Some {value:List<Item>}}
fn read(choice:Choice)->int {match choice {Some(p)=>{return (list_Item_get p.value 0).value}}}
shadow read {let values:List<Item> = (list_Item_new) (list_Item_push values Item {value:7}) assert (== (read Choice.Some {value:values}) 7)}
fn main()->int {let values:List<Item> = (list_Item_new) (list_Item_push values Item {value:9}) assert (== (read Choice.Some {value:values}) 9) return 0}
'''+shared.SHADOW)

    def test_distinct_primitive_list_arguments(self):
        self.execute('''union Box<T> {Some {value:T}}
fn number(box:Box<List<int>>)->int {match box {Some(p)=>{return (list_int_get p.value 0)}}}
shadow number {let values:List<int> = (list_int_new) (list_int_push values 7) assert (== (number Box<List<int>>.Some {value:values}) 7)}
fn text(box:Box<List<string>>)->string {match box {Some(p)=>{return (list_string_get p.value 0)}}}
shadow text {let values:List<string> = (list_string_new) (list_string_push values "kept") assert (== (text Box<List<string>>.Some {value:values}) "kept")}
fn main()->int {let numbers:List<int> = (list_int_new) let strings:List<string> = (list_string_new) (list_int_push numbers 7) (list_string_push strings "kept") assert (== (number Box<List<int>>.Some {value:numbers}) 7) assert (== (text Box<List<string>>.Some {value:strings}) "kept") return 0}
'''+shared.SHADOW)

    def test_imported_record_tuple_arguments_keep_owners(self):
        self.execute('''module "a.nano" as a
module "b.nano" as b
union Box<T> {Some {value:T}}
fn first(box:Box<(a.Item,int)>)->int {match box {Some(p)=>{let item:a.Item = p.value.0 return (a.read item)}}}
shadow first {assert (== (first Box<(a.Item,int)>.Some {value:((a.make),1)}) 7)}
fn second(box:Box<(b.Item,int)>)->int {match box {Some(p)=>{let item:b.Item = p.value.0 return (b.read item)}}}
shadow second {assert (== (second Box<(b.Item,int)>.Some {value:((b.make),1)}) 9)}
fn main()->int {assert (== (first Box<(a.Item,int)>.Some {value:((a.make),1)}) 7) assert (== (second Box<(b.Item,int)>.Some {value:((b.make),1)}) 9) return 0}
'''+shared.SHADOW, files={name+'.nano':f'pub struct Item {{value:int}}\npub fn make()->Item {{return Item {{value:{value}}}}}\nshadow make {{assert true}}\npub fn read(item:Item)->int {{return item.value}}\nshadow read {{assert (== (read (make)) {value})}}\n' for name,value in [('a',7),('b',9)]})
