"""I retain source-level opaque/carrier controls for all installed producers.

I use the owning SDK fixture's durable runner and exact selected-shadow audit.
Negative runtime programs have safe shadows; their deliberate absent loads run
only in separately supervised native executables, after compilation succeeds.
"""
import hashlib
import json
from pathlib import Path

COMPILERS = ('nanoc_c', 'nanoc_stage1', 'nanoc')


def program(body, declarations=''):
    return declarations + '\nfn main() -> int {\n' + body + '\nreturn 0\n}\nshadow main { assert (== (main) 0) }\n'


TUPLES = '''
fn pair(value:int)->(int,bool) { return (value,true) }
shadow pair { let result:(int,bool)=(pair 7) assert (== result.0 7) assert result.1 }
fn keep(value:( int, bool ))->bool { return value.1 }
shadow keep { assert (keep (7,true)) }
fn increment(value:(int,bool))->( int , bool ) { return ((+ value.0 1),value.1) }
shadow increment { let result:(int,bool)=(increment (7,true)) assert (== result.0 8) }
'''
CALLBACKS = '''
fn plus(value:int)->int { return (+ value 1) }
shadow plus { assert (== (plus 7) 8) }
fn minus(value:int)->int { return (- value 1) }
shadow minus { assert (== (minus 7) 6) }
fn select(value:fn( int )->int)->fn(int) -> int { return value }
shadow select { let callback:fn(int)->int=(select plus) assert (== (callback 7) 8) }
fn retain(value:fn(int)->int)->bool { return (== (value 7) 8) }
shadow retain { assert (retain plus) }
'''


def sources(directory):
    """I construct immutable named cases, not an alternate language parser."""
    directory=Path(directory);directory.mkdir()
    positive={}
    positive['tuple-operations']=program('''
let mut values:array<(int,bool)> =[]
let empty:(int,bool)=(array_pop values)
assert (== empty.0 0) assert (not empty.1)
set values (array_push values (pair 7))
set values (array_push values (pair 11))
(array_set values 0 (pair 13))
let first:(int,bool)=(array_get values 0)
assert (== first.0 13)
let mapped:array<( int , bool )> =(map values increment)
let chosen:array<(int,bool)> =(filter mapped keep)
let copied:array<(int,bool)> =(array_slice chosen 0 2)
let popped:(int,bool)=(array_pop copied)
assert (== popped.0 12)
let left:(int,bool)=(at copied 0)
assert (== left.0 14)
let repeated:array<(int,bool)> =(array_new 2 (pair 19))
let mut sum:int=0
for value in repeated { set sum (+ sum value.0) }
assert (== sum 38)
let nested:array<array<(int,bool)>> =[values]
let inside:array<(int,bool)> =(at nested 0)
let saved:(int,bool)=(at inside 1)
assert (== saved.0 11)
''',TUPLES)
    positive['callback-operations']=program('''
let mut values:array<fn(int)->int> =[]
set values (array_push values plus)
set values (array_push values minus)
let saved:fn( int ) -> int=(array_get values 0)
(array_set values 0 minus)
assert (== (saved 7) 8)
let selected:array<fn(int)->int> =(map values select)
let kept:array<fn( int )->int> =(filter [plus,minus] retain)
assert (== (array_length kept) 1)
let callback:fn(int)->int=(array_pop selected)
assert (== (callback 7) 6)
let repeated:array<fn(int)->int> =(array_new 2 plus)
let mut total:int=0
for item in repeated { set total (+ total (item 7)) }
assert (== total 16)
''',CALLBACKS)
    # These leaf types appear only in global annotations/initializers. No local
    # type declaration or typed helper parameter can accidentally prepare them.
    positive['global-only-empty']=program('''
assert (== (array_length global_pairs) 0)
assert (== (array_length global_callbacks) 0)
''','let global_pairs:array<(int,bool)> =[]\nlet global_callbacks:array<fn(int)->int> =[]\n')
    positive['global-only-values']=program('''
assert (== (at global_pairs 0).0 37)
assert (at global_pairs 0).1
assert (== ((at global_callbacks 0) 7) 8)
''','let global_pairs:array<(int,bool)> =[(37,true)]\nlet global_callbacks:array<fn(int)->int> =[plus]\n'+
        'fn plus(value:int)->int { return (+ value 1) }\nshadow plus { assert (== (plus 7) 8) }\n')
    positive['relocation-copy']=program("""
set moving [(37,true),(41,true)]
let values:array<(int,bool)>=(map moving grow)
assert (== (array_length values) 2)
assert (== (at values 0).0 37) assert (== (at values 1).0 41)
assert (== (array_length moving) 130)
""",'let mut moving:array<(int,bool)> = []\n'+"""
fn grow(value:(int,bool))->(int,bool) {
 let mut count:int=0
 while (< count 64) { set moving (array_push moving (99,false)) set count (+ count 1) }
 return value
}
shadow grow { set moving [] let value:(int,bool)=(grow (37,true)) assert (== value.0 37) assert value.1 }
""")
    positive['fresh-empty-pop']=program("""
let pairs:array<(int,bool)>=(array_new 0 (37,true))
let first:(int,bool)=(array_pop pairs)
assert (== first.0 0) assert (not first.1)
let callbacks:array<fn(int)->int>=(array_new 0 plus)
let ignored:fn(int)->int=(array_pop callbacks)
assert (== (array_length callbacks) 0)
""",CALLBACKS)
    positive['single-evaluation']=program("""
set source_calls 0 set index_calls 0
let value:(int,bool)=(array_get (source) (index))
assert (== value.0 37)
assert (== source_calls 1) assert (== index_calls 1)
""","""
let mut source_calls:int=0
let mut index_calls:int=0
fn source()->array<(int,bool)> { set source_calls (+ source_calls 1) return [(37,true)] }
shadow source { set source_calls 0 let values:array<(int,bool)>=(source) assert (== source_calls 1) assert (== (array_length values) 1) }
fn index()->int { set index_calls (+ index_calls 1) return 0 }
shadow index { set index_calls 0 assert (== (index) 0) assert (== index_calls 1) }
""")
    positive['ordinary-empty-globals']=program("""
assert (== (array_length global_strings) 0)
assert (== (array_length global_ints) 0)
assert (== (array_length mutable_strings) 0)
set mutable_strings (array_push mutable_strings "retained")
assert (== (at mutable_strings 0) "retained")
# I leave the next main invocation the same observable empty state, without
# replacing the global carrier or calling any compiler registry reset helper.
let removed:string=(array_pop mutable_strings)
assert (== removed "retained")
""","""
let global_strings:array<string> = []
let global_ints:array<int> = []
let mut mutable_strings:array<string> = []
""")
    positive['nonliteral-globals']=program("""
assert (== indirect 37) assert (== called 38) assert (== changed 39)
assert (== global_pair.first 37) assert (== global_pair.second 38)
assert (== global_tuple.0 37) assert (== global_tuple.1 38)
assert (== (at ordinary_values 0) 37) assert (== (at ordinary_values 1) 38)
assert (== (callback 41) 42) assert computed_flag assert (== computed_text "ab")
assert (== computed_int 75)
match global_box { Value(payload)=>{ assert (== payload.value 37) } }
""","""
struct InitialPair { first:int, second:int }
union Box<T> { Value { value:T } }
fn next(value:int)->int { return (+ value 1) }
shadow next { assert (== (next 41) 42) }
let literal:int=37
let indirect:int=literal
let called:int=(next indirect)
let mut changed:int=(next called)
let global_pair:InitialPair=InitialPair { first:indirect, second:called }
let global_tuple:(int,int)=(global_pair.first,global_pair.second)
let ordinary_values:array<int> = [global_pair.first,global_pair.second]
let callback:fn(int)->int=next
let computed_flag:bool=(== called 38)
let computed_text:string=(str_concat "a" "b")
let computed_int:int=(+ indirect called)
let global_box:Box<int> = Box<int>.Value { value:indirect }
""")
    # Actual main-scope definition admission is preserved: these names already
    # permit declarations. Reserved declarations are not silently reclassified.
    for name in ('array_push','array_pop','filter','array_filter','array_map','array_fold','array_remove_at'):
        positive['declared-'+name]=program(f'''
let value:(int,bool)=({name} (37,true))
assert (== value.0 37) assert value.1
''',f'fn {name}(value:(int,bool))->(int,bool) {{ return value }}\n'
           f'shadow {name} {{ let value:(int,bool)=({name} (37,true)) assert (== value.0 37) }}\n')
    for name in ('array_new','array_push','array_set','array_get','at','array_pop','map','array_map','filter','array_filter','reduce','array_fold','array_slice','array_remove_at'):
        positive['local-'+name]=program(f'''
let {name}:fn((int,bool))->(int,bool)=identity
let value:(int,bool)=({name} (37,true))
assert (== value.0 37) assert value.1
''','fn identity(value:(int,bool))->(int,bool) { return value }\n'
           'shadow identity { let value:(int,bool)=(identity (37,true)) assert (== value.0 37) }\n')
    # Identical basename, distinct explicit public labels, distinct physical
    # owners. No foreign pointer casts manufacture identity in these cases.
    for label in ('Left','Right'):
        leaf=directory/label;leaf.mkdir()
        (leaf/'owner.nano').write_text(f'''module {label}Opaque
opaque type Handle
pub fn zero()->Handle {{ return 0 }}
shadow zero {{ assert (== (zero) 0) }}
pub fn valid(value:Handle)->bool {{ return (== value 0) }}
shadow valid {{ assert (valid (zero)) }}
''')
    imports='module "./Left/owner.nano" as Left\nmodule "./Right/owner.nano" as Right\n'
    same='module "./Left/owner.nano" as Again\n'
    positive['opaque-origins']=program('''
let left:Left.Handle=(Left.zero)
let same:Again.Handle=left
let right:Right.Handle=(Right.zero)
assert (Again.valid same) assert (Left.valid left) assert (Right.valid right)
let handles:array<Again.Handle> =[left]
let from_array:Left.Handle=(at handles 0)
assert (Left.valid from_array)
''',imports+same)
    positive['opaque-origins-reversed']=positive['opaque-origins'].replace(imports,
        'module "./Right/owner.nano" as Right\nmodule "./Left/owner.nano" as Left\n')
    positive['opaque-map-composition']=program('''
let handles:array<Left.Handle> =[(Left.zero)]
let pairs:array<(Again.Handle,int)> =(map handles wrap)
let out:array<Left.Handle> =(map pairs unwrap)
assert (Left.valid (at out 0))
let chosen:array<(Left.Handle, int)> =(filter pairs valid_pair)
assert (== (array_length chosen) 1)
''',imports+same+'''
fn wrap(value:Again.Handle)->(Left.Handle,int) { return (value,37) }
shadow wrap { let pair:(Left.Handle,int)=(wrap (Again.zero)) assert (== pair.1 37) }
fn unwrap(value:( Again.Handle , int ))->Left.Handle { return value.0 }
shadow unwrap { assert (Left.valid (unwrap ((Again.zero),37))) }
fn valid_pair(value:(Left.Handle,int))->bool { return (== value.1 37) }
shadow valid_pair { assert (valid_pair ((Left.zero),37)) }
''')
    positive['nested-generic-opaque']=program("""
let first:Box<Left.Handle>=Box<Left.Handle>.Value { value:(Left.zero) }
let second:Box<Right.Handle>=Box<Right.Handle>.Value { value:(Right.zero) }
let nested:Box<Box<Left.Handle>>=Box<Box<Left.Handle>>.Value { value:first }
match first { Value(payload)=>{ assert (Left.valid payload.value) } }
match second { Value(payload)=>{ assert (Right.valid payload.value) } }
match nested { Value(outer)=>{ match outer.value { Value(inner)=>{ assert (Again.valid inner.value) } } } }
let paired:(Left.Handle,fn(Left.Handle)->bool)=((Left.zero),Left.valid)
let callback:fn(Again.Handle)->bool=paired.1
assert (callback paired.0)
""",imports+same+'union Box<T> { Value { value:T } }\n')
    positive['source-prefix-collision']=program("""
let opaque_value:Left.Handle=(Left.zero)
let value:(Left.Handle,int)=(opaque_value,37)
assert (== value.1 (__nano_opaque_0_probe))
""",imports+'fn __nano_opaque_0_probe()->int { return 37 }\nshadow __nano_opaque_0_probe { assert (== (__nano_opaque_0_probe) 37) }\n')
    # I observe actual native tags and widths through an ordinary C provider;
    # callback width is sizeof an actual function-pointer type, never void*.
    probe=directory/'carrier observer';probe.mkdir()
    (probe/'shape.h').write_text('#include "runtime/dyn_array.h"\nint64_t sdk_tuple_shape(DynArray*);\nint64_t sdk_callback_shape(DynArray*);\n')
    (probe/'shape.c').write_text("""#include "shape.h"
#include <assert.h>
#include <stdbool.h>
#include <string.h>
typedef struct { int64_t first; bool second; } Pair;
typedef int64_t (*Callback)(int64_t);
const uint32_t sdk_tuple_shape__nano_array_abi = 2u;
const uint32_t sdk_callback_shape__nano_array_abi = 2u;
int64_t sdk_tuple_shape(DynArray *a) {
 assert(a && a->elem_type == ELEM_STRUCT && a->elem_size == sizeof(Pair) && a->length == 1);
 Pair value; memcpy(&value, dyn_array_get_struct(a,0), sizeof value);
 assert(value.first == 37 && value.second); return 37;
}
int64_t sdk_callback_shape(DynArray *a) {
 assert(a && a->elem_type == ELEM_STRUCT && a->elem_size == sizeof(Callback) && a->length == 1);
 Callback value; memcpy(&value, dyn_array_get_struct(a,0), sizeof value);
 assert(value && value(7) == 8); return 37;
}
""")
    (probe/'module.json').write_text(json.dumps(dict(name='SdkCarrierObserver',headers=['shape.h'],c_sources=['shape.c'])))
    (probe/'observer.nano').write_text('module SdkCarrierObserver\npub extern fn sdk_tuple_shape(value:array<(int,bool)>)->int\npub extern fn sdk_callback_shape(value:array<fn(int)->int>)->int\n')
    positive['actual-tag-width']=program("""
unsafe {
 assert (== (Observer.sdk_tuple_shape [(37,true)]) 37)
 assert (== (Observer.sdk_callback_shape [plus]) 37)
}
""",'module "./carrier observer/observer.nano" as Observer\n'+CALLBACKS)
    negative={
        'unknown-qualifier':program('let value:Missing.Handle=(Left.zero)',imports),
        'unknown-member':program('let value:Left.Missing=(Left.zero)',imports),
        'cross-owner-let':program('let value:Right.Handle=(Left.zero)',imports),
        'cross-owner-argument':program('assert (Right.valid (Left.zero))',imports),
        'cross-owner-return':program('let value:Right.Handle=(wrong)',imports+
            'fn wrong()->Right.Handle { return (Left.zero) }\nshadow wrong { assert (Right.valid (wrong)) }\n'),
        'cross-owner-array':program('let values:array<Right.Handle> =[(Left.zero)]',imports),
        'cross-owner-tuple':program('let value:(Right.Handle,int)=((Left.zero),37)',imports),
        'duplicate-local':program('', 'opaque type Handle\nopaque type Handle\n'),
        'record-kind':program('let value:Handle=(Left.zero)',imports+'struct Handle { value:int }\n'),
        'callback-result':program('let value:fn(int)->bool=plus',CALLBACKS),
    }
    runtime={}
    for index in (-1,1):
        runtime['absent-'+str(index).replace('-','negative')]='''
fn load(index:int)->int {
 let values:array<(int,bool)> =[(37,true)]
 let value:(int,bool)=(at values index)
 return value.0
}
shadow load { assert (== (load 0) 37) }
'''+f'fn main()->int {{ return (load {index}) }}\nshadow main {{ assert (== (load 0) 37) }}\n'
    for index in (-1,1):
        runtime['absent-callback-'+str(index).replace('-','negative')]=CALLBACKS+"""
fn load(index:int)->int {
 let values:array<fn(int)->int> = [plus]
 let callback:fn(int)->int=(at values index)
 return (callback 7)
}
shadow load { assert (== (load 0) 8) }
"""+f'fn main()->int {{ return (load {index}) }}\nshadow main {{ assert (== (load 0) 8) }}\n'
    runtime['shrink-map']='''
let mut shared:array<(int,bool)> =[]
fn shrink(value:(int,bool))->(int,bool) {
 let ignored:(int,bool)=(array_pop shared)
 return value
}
shadow shrink {
 set shared [(7,true)]
 let value:(int,bool)=(shrink (37,true))
 assert (== value.0 37) assert (== (array_length shared) 0)
}
fn main()->int {
 set shared [(7,true),(11,true)]
 let values:array<(int,bool)> =(map shared shrink)
 return (array_length values)
}
shadow main {
 set shared [(7,true)]
 let values:array<(int,bool)> =(map shared shrink)
 assert (== (array_length values) 1)
 assert (== (at values 0).0 7)
}
'''
    runtime['shrink-filter']=runtime['shrink-map'].replace(
        'fn shrink(value:(int,bool))->(int,bool)', 'fn shrink(value:(int,bool))->bool').replace(
        ' return value\n}', ' return value.1\n}').replace(
        'let value:(int,bool)=(shrink (37,true))\n assert (== value.0 37)',
        'let value:bool=(shrink (37,true))\n assert value').replace('(map shared shrink)', '(filter shared shrink)')
    runtime['shrink-iteration']="""
let mut shared:array<(int,bool)> = []
fn consume()->int {
 let mut result:int=0
 for value in shared { let removed:(int,bool)=(array_pop shared) set result (+ result value.0) }
 return result
}
shadow consume { set shared [(7,true)] assert (== (consume) 7) }
fn main()->int { set shared [(7,true),(11,true)] return (consume) }
shadow main { set shared [(7,true)] assert (== (consume) 7) }
"""
    rows=[]
    for kind,cases in (('positive',positive),('refusal',negative),('runtime-refusal',runtime)):
        for name,text in cases.items():
            source=directory/(name+'.nano');source.write_text(text.replace('>=', '> ='))
            rows.append(dict(name=name,kind=kind,path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    assets={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
                       mode=p.stat().st_mode & 0o7777)
            for p in sorted(directory.rglob('*')) if p.is_file()}
    (directory/'source-inputs.json').write_text(json.dumps(assets,indent=2)+'\n')
    (directory/'cases.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def run(test):
    cases=sources(test.outside/'opaque carrier cases')
    assets=json.loads((test.outside/'opaque carrier cases/source-inputs.json').read_text())
    summary=[]
    for compiler in COMPILERS:
        for row in cases:
            label='opaque-'+compiler+'-'+row['name'];source=Path(row['path'])
            if row['kind']=='positive':
                test.compile_installed(label,compiler,source)
            elif row['kind']=='refusal':
                output=test.outside/(label+'-program');output.write_bytes(b'I preserve the previous output.\n')
                out,err,status=test.command(label+'-refusal',
                    [test.generation/'bin'/compiler,source,'-o',output,'--keep-c'],expected=(1,),timeout=900)
                test.assertEqual(output.read_bytes(),b'I preserve the previous output.\n')
                test.assertTrue(out or err,'I require an actual refusal diagnostic')
                if row['name'] in ('unknown-qualifier','unknown-member'):
                    test.assertIn(b'I cannot resolve a qualified type in this source:',out+err)
                else:
                    diagnostic=(out+err).lower()
                    test.assertTrue(any(word in diagnostic for word in (b'type',b'opaque',b'declaration',b'callable')),diagnostic)
                    test.assertNotIn(b'syntax error',diagnostic)
                    test.assertNotIn(b'parse error',diagnostic)
                test.assertNotIn(b'I cannot load an absent array element',out+err)
            else:
                # compile_installed retains the same complete shadow/command
                # proof while the separate native execution has an exact trap.
                test.compile_installed(label,compiler,source,
                    run_expected=(-6,),run_diagnostic=b'I cannot load an absent array element\n')
            summary.append(dict(compiler=compiler,**row))
    for name,identity in assets.items():
        path=Path(name)
        test.assertEqual(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
                              mode=path.stat().st_mode & 0o7777),identity)
    (test.work/'opaque-carrier-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
