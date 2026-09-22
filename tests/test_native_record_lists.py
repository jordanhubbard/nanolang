"""I qualify corrected native list providers separately from NanoISA admission.

I retain every command before assertions through the existing bounded supervisor.
The original eighteen-method GenericRecordLists suite remains an independent gate.
"""
from pathlib import Path
import hashlib
import json
import os
import shlex
import sys
import tempfile
import unittest
from tests import test_generic_record_lists as retained

ROOT = Path(__file__).resolve().parents[1]


class NativeRecordLists(unittest.TestCase):
    command = classmethod(retained.GenericRecordLists.command.__func__)
    products = classmethod(retained.GenericRecordLists.products.__func__)

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-native-record-lists-', dir=os.environ.get('NANO_LIST_REPORT_DIR')))
        print('I retain native list artifacts at', cls.work, flush=True)
        (cls.work / 'temporary').mkdir()
        cls.sequence = 0
        cls.cc = shlex.split(os.environ.get('NANO_LIST_CC', 'cc'))
        cls.flags = shlex.split(os.environ.get('NANO_LIST_CFLAGS', ''))
        cls.links = shlex.split(os.environ.get('NANO_LIST_LDFLAGS', ''))
        cls.native_cc = shlex.split(os.environ['NANO_NATIVE_LIST_CC'])
        cls.native_flags = shlex.split(os.environ.get('NANO_NATIVE_LIST_CFLAGS', ''))
        cls.native_links = shlex.split(os.environ.get('NANO_NATIVE_LIST_LDFLAGS', ''))
        if not cls.native_cc or not cls.cc:
            raise AssertionError('I require actual selected native compiler commands.')
        cls.inputs = [ROOT / 'src/runtime/native_record_list.h', ROOT / 'src/runtime/list_capacity.h',
                      ROOT / 'tests/test_native_record_list_storage.c', Path(__file__).resolve(),
                      *[ROOT / 'bin' / name for name in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2')]]
        if any(not path.is_file() for path in cls.inputs):
            raise AssertionError('I require all three freshly prepared source producers.')
        cls.before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.inputs}
        (cls.work / 'fixture-inputs-before.json').write_text(json.dumps(cls.before, indent=2) + '\n')
        (cls.work / 'selected-config.json').write_text(json.dumps(dict(cc=cls.cc, flags=cls.flags,
            links=cls.links, native_cc=cls.native_cc, native_flags=cls.native_flags,
            native_links=cls.native_links, python=sys.executable), indent=2) + '\n')
        cls.storage = cls.work / 'native-storage'
        cls.command('storage-build', [*cls.cc, *cls.flags, '-std=c99', '-Wall', '-Wextra', '-Werror',
            ROOT / 'tests/test_native_record_list_storage.c', *cls.links, '-o', cls.storage])
        # I select optimization through the actual compiler hook, preserving all
        # original arguments and recording the final invocation. No shell eval.
        cls.wrapper = cls.work / 'native-cc.py'
        cls.wrapper.write_text('''import json, os, pathlib, sys, time
config = json.loads(os.environ['NATIVE_LIST_TOOL_CONFIG'])
linking = not any(flag in sys.argv[1:] for flag in ('-c', '-E', '-S'))
argv = config['cc'] + sys.argv[1:] + config['flags'] + (config['links'] if linking else []) + ['-Wall', '-Wextra', '-Werror', config['optimization']]
root = pathlib.Path(config['reports'])
(root / ('native-argv-' + str(time.time_ns()) + '-' + str(os.getpid()) + '.json')).write_text(json.dumps(dict(argv=argv, cwd=os.getcwd()), indent=2) + '\\n')
os.execvp(argv[0], argv)
''')

    @classmethod
    def tearDownClass(cls):
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.inputs}
        (cls.work / 'fixture-inputs-after.json').write_text(json.dumps(after, indent=2) + '\n')
        if after != cls.before:
            raise AssertionError('My selected native list fixture inputs changed.')

    def test_shared_storage_and_allocation_failures(self):
        out, _ = self.command('storage-controls', [self.storage])
        self.assertIn(b'allocation positions in prefix and transient modes', out)
        self.assertIn(b'native record-list assertions', out)
        for mode in ('real-index', 'real-capacity', 'real-null', 'real-oom'):
            _, err = self.command(mode, [self.storage, mode], expected=(1,))
            self.assertIn(b'I cannot complete this list operation', err)

    def native_routes(self, name, source, runtime_failure=False, extra_links=(), expected_stdout=None):
        if isinstance(source, Path):
            path = source
        else:
            path = self.work / (name + '.nano')
            path.write_text(source)
        (self.work / (name + '-source.json')).write_text(json.dumps(dict(path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2) + '\n')
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            for optimization in ('-O0', '-O2'):
                label = name + '-' + compiler + optimization
                output = self.work / label
                output.write_bytes(retained.SENTINEL)
                config = dict(cc=self.native_cc, flags=self.native_flags, links=[*self.native_links, *map(str, extra_links)],
                              optimization=optimization, reports=str(self.work))
                extra = dict(NATIVE_LIST_TOOL_CONFIG=json.dumps(config),
                             NANO_CC=shlex.join([sys.executable, str(self.wrapper)]))
                previous = set(self.work.glob('native-argv-*.json'))
                self.command(label + '-build', [ROOT / 'bin' / compiler, path, '-o', output, '--keep-c'],
                             timeout=300, extra=extra)
                self.assertNotEqual(output.read_bytes(), retained.SENTINEL)
                invocations = sorted(set(self.work.glob('native-argv-*.json')) - previous)
                self.assertTrue(invocations, 'I require the actual selected compiler hook to run.')
                for invocation in invocations:
                    argv = json.loads(invocation.read_text())['argv']
                    self.assertEqual(argv[-1], optimization)
                    self.assertIn('-Werror', argv)
                out, err = self.command(label + '-run', [output], timeout=15,
                                      expected=(1,) if runtime_failure else (0,))
                if runtime_failure:
                    self.assertIn(b'I cannot complete this list operation', err)
                if expected_stdout is not None:
                    self.assertEqual(out, expected_stdout)

    def test_signed_string_length_results(self):
        self.native_routes('signed-string-length', '''fn below(text:string, width:int) -> int {
 return (- (str_length text) width)
}
shadow below { assert (== (below "a" 3) -2) }
fn once() -> string { (print "x") return "ab" }
shadow once { assert (== (str_length (once)) 2) }
fn main() -> int {
 assert (== (str_length "") 0)
 assert (== (str_length "éx") 3)
 assert (< (- (str_length "a") 3) 0)
 assert (not (< (str_length "a") -1))
 assert (== (/ (- (str_length "a") 4) 2) -1)
 assert (== (below "a" 3) -2)
 assert (== (- (str_length (once)) 3) -1)
 return 0
}
''', expected_stdout=b'x')

    def test_native_operations_order_and_growth(self):
        # I retain the existing trace/alias/iteration program and all assertions.
        source = (ROOT / 'tests/fixtures/evaluator_lists/mutations.nano').read_text()
        self.native_routes('native-mutations', source)
        self.native_routes('native-initializer-publication', '''struct Item { value:int }
fn bump(value:int)->int { (print "x") return (+ value 1) }
shadow bump { assert (== (bump 40) 41) }
fn inc(value:int)->int { return (+ value 1) }
shadow inc { assert (== (inc 4) 5) }
extern fn __nl_let_initializer_8_0()->int
fn __nl_let_initializer_0()->int { return 19 }
shadow __nl_let_initializer_0 { assert (== (__nl_let_initializer_0) 19) }
fn main()->int {
 let __nl_let_initializer_1:int = 23
 let value:int = 40
 let text:string = "outer"
 let item:Item = Item{value:7}
 let values:array<int> = [8,9]
 let pair:(int,string) = (11,"pair")
 let callback:fn(int)->int = inc
 let byte:int = 257
 if true {
  let value:int = (bump value)
  let __nl_let_initializer_2:int = 29
  assert (== __nl_let_initializer_2 29)
  let text:string = text
  let item:Item = item
  let values:array<int> = values
  let pair:(int,string) = pair
  let callback:fn(int)->int = callback
  let byte:u8 = byte
  assert (== value 41)
  assert (== text "outer")
  assert (== item.value 7)
  assert (== (at values 1) 9)
  assert (== pair.0 11)
  assert (== pair.1 "pair")
  assert (== (callback 8) 9)
  assert (== byte 1)
 }
 if true { let inc:fn(int)->int = inc assert (== (inc 6) 7) }
 assert (== value 40)
 assert (== byte 257)
 assert (== __nl_let_initializer_1 23)
 assert (== (__nl_let_initializer_0) 19)
 return 0
}
''', expected_stdout=b'x')

    def test_native_discovery_and_copied_results(self):
        self.native_routes('native-discovery', '''struct Item { value:int, text:string }
struct Holder { values:List<Item> }
union Held<T> { Items { values:List<T> } }
fn make() -> List<Item> { return (list_Item_new) }
shadow make { let xs=(make) assert (list_Item_is_empty xs) (list_Item_free xs) }
fn take(xs:List<Item>) -> Item { return (list_Item_remove xs 0) }
shadow take { let xs=(make) (list_Item_push xs Item{value:3,text:"three"}) assert (== (take xs).value 3) (list_Item_free xs) }
fn main() -> int {
 let xs=(make)
 (list_Item_push xs Item{value:7,text:"seven"})
 let holder:Holder=Holder{values:xs}
 let held:Held<Item> =Held.Items{values:xs}
 match held { Items(payload) => { assert (== (list_Item_length payload.values) 1) } }
 let removed=(take holder.values)
 (list_Item_clear xs)
 (list_Item_free xs)
 assert (== removed.value 7)
 assert (== removed.text "seven")
 return 0
}
shadow main { assert (== (main) 0) }
''')
        self.native_routes('shadow-only-discovery', '''struct Item { value:int }
fn main()->int{return 0}
shadow main { let xs=(list_Item_new) (list_Item_push xs Item{value:5}) assert (== (list_Item_pop xs).value 5) (list_Item_free xs) }
''')

        self.native_routes('native-nested-list-projections', '''struct Item { value:int }
union Inner<T> { Items { values:List<T> } }
union Outer<T> { Wrapped { inner:Inner<T> } }
struct Holder { value:Outer<Item> }
fn build(values:List<Item>)->Outer<Item> { return Outer<Item>.Wrapped { inner:Inner<Item>.Items { values:values } } }
shadow build { let xs=(list_Item_new) (list_Item_push xs Item{value:13}) match (build xs) { Wrapped(outer)=>{ match outer.inner { Items(inner)=>{ assert (== (list_Item_get inner.values 0).value 13) } } } } (list_Item_free xs) }
fn main()->int {
 let xs=(list_Item_new)
 (list_Item_push xs Item{value:13})
 let holder:Holder=Holder{value:(build xs)}
 match holder.value { Wrapped(outer)=>{
  let alias=outer
  let inner=alias.inner
  match inner { Items(payload)=>{
   let saved=payload.values
   assert (== (list_Item_length saved) 1)
   assert (== (list_Item_get saved 0).value 13)
  } }
 } }
 match (build xs) { Wrapped(outer)=>{ match outer.inner { Items(payload)=>{ assert (== (list_Item_length payload.values) 1) } } } }
 match Outer<Item>.Wrapped{inner:Inner<Item>.Items{values:xs}} { Wrapped(outer)=>{ match outer.inner { Items(payload)=>{ assert (== (list_Item_get payload.values 0).value 13) } } } }
 let projected=(match Outer<Item>.Wrapped{inner:Inner<Item>.Items{values:xs}} {
  Wrapped(outer)=>{ let alias=outer
   if true { let alias=alias match alias.inner {Items(payload)=>{assert (== (list_Item_length payload.values) 1)}} }
   (match alias.inner {Items(payload)=>{(list_Item_get payload.values 0).value}})
  }
 })
 assert (== projected 13)
 (list_Item_free xs)
 return 0
}
shadow main { assert (== (main) 0) }
''')

        self.native_routes('native-contextual-nested-constructors', '''union Inner<T> { Items { value:T } }
union Outer<T> { Wrapped { inner:Inner<T> } }
fn build()->Outer<int> { return Outer.Wrapped{inner:Inner.Items{value:17}} }
shadow build { match (build) { Wrapped(outer)=>{ match outer.inner { Items(inner)=>{assert (== inner.value 17)} } } } }
fn main()->int {
 let value:Outer<int> = Outer.Wrapped{inner:Inner.Items{value:23}}
 match value { Wrapped(outer)=>{ match outer.inner { Items(inner)=>{assert (== inner.value 23)} } } }
 match (build) { Wrapped(outer)=>{ match outer.inner { Items(inner)=>{assert (== inner.value 17)} } } }
 return 0
}
shadow main { assert (== (main) 0) }
''')

        self.native_routes('native-selected-payload-carriers', '''union Box<T> { Some { value:T }, None {} }
fn main()->int {
 let number:Box<int> = Box<int>.Some{value:31}
 let text:Box<string> = Box<string>.Some{value:"text"}
 match number { Some(payload)=>{
  let alias=(cond (true payload) (else payload))
  let pair=(alias,payload)
  assert (== pair.0.value 31)
  assert (== pair.1.value 31)
 } None(empty)=>{ let alias=empty } }
 match text { Some(payload)=>{ let alias=payload assert (== alias.value "text") } None(empty)=>{} }
 let absent:Box<int> = Box<int>.None{}
 match absent { Some(payload)=>{ assert false } None(empty)=>{ let alias=empty } }
 return 0
}
shadow main { assert (== (main) 0) }
''')

    def test_native_imported_owners_and_long_names(self):
        directory = self.work / 'owners'; directory.mkdir()
        for module, value in (('Left', 11), ('Right', 22)):
            (directory / (module + '.nano')).write_text(f'''module {module}
pub struct Item {{ value:int }}
pub fn sample()->int{{let xs:List<Item> =(list_Item_new) (list_Item_push xs Item{{value:{value}}}) let value=(list_Item_pop xs) (list_Item_free xs) return value.value}}
shadow sample {{ assert (== (sample) {value}) }}
''')
        self.native_routes('native-imported-owners', f'''module "{directory / 'Left.nano'}" as left
module "{directory / 'Right.nano'}" as right
fn main()->int{{assert (== (left.sample) 11) assert (== (right.sample) 22) return 0}}
shadow main {{ assert (== (main) 0) }}
''')
        (directory / 'Payloads.nano').write_text('''module Payloads
pub struct Item { value:int }
pub union Mixed<T> { Both { fixed:List<Item>, supplied:List<T> } }
pub union Nested<T> { Wrapped { inner:Mixed<T> } }
pub fn fixed()->List<Item>{let xs=(list_Item_new) (list_Item_push xs Item{value:17}) return xs}
shadow fixed {let xs=(fixed) assert (== (list_Item_get xs 0).value 17) (list_Item_free xs)}
pub fn inspect(value:List<Item>)->int{return (list_Item_get value 0).value}
shadow inspect {let xs=(fixed) assert (== (inspect xs) 17) (list_Item_free xs)}
pub fn release(value:List<Item>)->void{(list_Item_free value)}
shadow release {(release (fixed))}
pub fn wrap(value:List<Item>)->Nested<Item>{return Nested<Item>.Wrapped{inner:Mixed<Item>.Both{fixed:value,supplied:value}}}
shadow wrap {let xs=(fixed) match (wrap xs){Wrapped(outer)=>{match outer.inner{Both(payload)=>{assert (== (inspect payload.supplied) 17)}}}} (release xs)}
''')
        self.native_routes('native-imported-list-payloads', f'''module "{directory / 'Payloads.nano'}" as p
struct Item {{ value:int }}
fn main()->int{{
 let own=(list_Item_new) (list_Item_push own Item{{value:29}})
 let fixed=(p.fixed)
 let value:p.Nested<Item> =p.Nested<Item>.Wrapped{{inner:p.Mixed<Item>.Both{{fixed:fixed,supplied:own}}}}
 match value {{ Wrapped(outer)=>{{let inner=outer.inner match inner {{Both(payload)=>{{
  assert (== (p.inspect payload.fixed) 17)
  assert (== (list_Item_get payload.supplied 0).value 29)
 }}}}}}}}
 match (p.wrap fixed) {{Wrapped(outer)=>{{match outer.inner{{Both(payload)=>{{assert (== (p.inspect payload.supplied) 17)}}}}}}}}
 match p.Nested<Item>.Wrapped{{inner:p.Mixed<Item>.Both{{fixed:fixed,supplied:own}}}} {{Wrapped(outer)=>{{
  let alias=outer
  if true {{ let alias=alias
   match alias.inner{{Both(payload)=>{{
    let copy=payload
    assert (== (p.inspect copy.fixed) 17)
    assert (== (list_Item_get copy.supplied 0).value 29)
   }}}}
  }}
 }}}}
 (p.release fixed) (list_Item_free own) return 0
}}
shadow main{{assert (== (main) 0)}}
''')
        (directory / 'Callbacks.nano').write_text('''module Callbacks
pub struct Item { value:int }
pub union Mixed<T> {
 Run { callback:fn(Item,T)->T, factory:fn()->fn(Item,T)->T }
}
pub fn fixed()->Item { return Item{value:17} }
shadow fixed { assert (== (fixed).value 17) }
''')
        callback_source = '''module "@CALLBACKS@" as p
struct Item { value:int }
fn combine(fixed:p.Item, own:Item)->Item { return Item{value:(+ fixed.value own.value)} }
shadow combine { assert (== (combine (p.fixed) Item{value:29}).value 46) }
fn factory()->fn(p.Item,Item)->Item { return combine }
shadow factory { let callback=(factory) assert (== (callback (p.fixed) Item{value:29}).value 46) }
fn invoke(callback:fn(p.Item,Item)->Item, fixed:p.Item, own:Item)->Item { return (callback fixed own) }
shadow invoke { assert (== (invoke combine (p.fixed) Item{value:29}).value 46) }
fn extract(value:p.Mixed<Item>)->fn(p.Item,Item)->Item {
 return match value { Run(payload)=>payload.callback }
}
shadow extract {
 let value:p.Mixed<Item> =p.Mixed<Item>.Run{callback:combine,factory:factory}
 let callback=(extract value)
 assert (== (callback (p.fixed) Item{value:29}).value 46)
}
fn choose(flag:bool, value:p.Mixed<Item>)->fn(p.Item,Item)->Item {
 let callback=(match flag { true=>{return combine} false=>{(extract value)} })
 return callback
}
shadow choose {
 let value:p.Mixed<Item> =p.Mixed<Item>.Run{callback:combine,factory:factory}
 let first=(choose true value) let second=(choose false value)
 assert (== (first (p.fixed) Item{value:29}).value 46)
 assert (== (second (p.fixed) Item{value:29}).value 46)
}
fn main()->int {
 let value:p.Mixed<Item> =p.Mixed<Item>.Run{callback:combine,factory:factory}
 match value { Run(payload)=>{
  let mut alias=payload.callback
  assert (== (alias (p.fixed) Item{value:29}).value 46)
  assert (== (invoke alias (p.fixed) Item{value:29}).value 46)
  set alias (cond (true payload.callback) (else alias))
  assert (== (alias (p.fixed) Item{value:29}).value 46)
  let selected=(cond (false alias) (else payload.callback))
  assert (== (selected (p.fixed) Item{value:29}).value 46)
  let matched=(match true { true=>alias false=>payload.callback })
  assert (== (matched (p.fixed) Item{value:29}).value 46)
  let maker=payload.factory
  let nested=(maker)
  assert (== (nested (p.fixed) Item{value:29}).value 46)
 }}
 let returned=(extract value)
 assert (== (returned (p.fixed) Item{value:29}).value 46)
 return 0
}
shadow main { assert (== (main) 0) }
'''.replace('@CALLBACKS@', str(directory / 'Callbacks.nano'))
        self.native_routes('native-imported-callable-payloads', callback_source)
        # I check rejection before native publication, never execute these outputs.
        for case, original, replacement in (
            ('argument', '(alias (p.fixed) Item{value:29})', '(alias Item{value:29} (p.fixed))'),
            ('returned', 'fn extract(value:p.Mixed<Item>)->fn(p.Item,Item)->Item',
                         'fn extract(value:p.Mixed<Item>)->fn(Item,p.Item)->Item'),
        ):
            path = self.work / ('callable-owner-' + case + '.nano')
            path.write_text(callback_source.replace(original, replacement, 1))
            output = self.work / ('callable-owner-' + case)
            output.write_bytes(retained.SENTINEL)
            out, err = self.command('callable-owner-' + case,
                [ROOT / 'bin/nanoc_c', path, '-o', output, '--keep-c'],
                expected=tuple(range(1, 126)), timeout=300)
            self.assertEqual(output.read_bytes(), retained.SENTINEL)
            self.assertIn(b'TYPE MISMATCH', out + err)
            self.assertNotIn(b'C compilation failed', out + err)
        (directory / 'Tuples.nano').write_text('''module Tuples
pub struct Item { value:int }
pub union Bundle<T> { Stored { payload:(Item,T,fn(Item,T)->T) } }
pub fn fixed()->Item { return Item{value:17} }
shadow fixed { assert (== (fixed).value 17) }
''')
        tuple_source = (ROOT / 'tests/fixtures/evaluator_lists/native_tuple_callbacks.nano').read_text().replace(
            '@TUPLES@', str(directory / 'Tuples.nano'))
        self.native_routes('native-complete-tuple-callbacks', tuple_source)
        for case, original, replacement in (
            ('tuple-owner', 'Box{pair:own,callback:combine}', 'Box{pair:fixed,callback:combine}'),
            ('tuple-callback-owner', 'fn extract(value:p.Bundle<Item>)->fn(p.Item,Item)->Item',
                                     'fn extract(value:p.Bundle<Item>)->fn(Item,p.Item)->Item'),
        ):
            self.assertIn(original, tuple_source)
            path = self.work / (case + '.nano')
            path.write_text(tuple_source.replace(original, replacement, 1))
            output = self.work / case
            output.write_bytes(retained.SENTINEL)
            out, err = self.command(case, [ROOT / 'bin/nanoc_c', path, '-o', output, '--keep-c'],
                                    expected=tuple(range(1, 126)), timeout=300)
            self.assertEqual(output.read_bytes(), retained.SENTINEL)
            self.assertIn(b'TYPE MISMATCH', out + err)
            self.assertNotIn(b'C compilation failed', out + err)
        name = 'Record_' + 'long_' * 24 + 'End'
        self.assertGreater(len(name), 64)
        self.assertLess(len(name), 250)
        self.native_routes('native-long-record', f'''struct {name} {{ value:int }}
fn main()->int{{let xs:List<{name}> =(list_{name}_new) (list_{name}_push xs {name}{{value:19}}) assert (== (list_{name}_remove xs 0).value 19) (list_{name}_free xs) return 0}}
shadow main {{ assert (== (main) 0) }}
''')

    def test_native_unchanged_lexer_and_schema_index(self):
        self.native_routes('native-original-lexer', ROOT / 'tests/token_value_bytes.nano',
            expected_stdout=b'0:4:4\n1:1:2\n2:3:8\n3:3:6\n4:3:2\n5:70:3\n6:7:0\n7:8:0\n8:0:0\n')
        self.native_routes('native-schema-index', '''import "src_nano/compiler/lexer.nano"
fn main()->int {
 let xs:List<LexerToken> =(list_LexerToken_new)
 (list_LexerToken_push xs LexerToken{token_type:3,value:"x",line:1,column:1,value_bytes:1})
 (list_LexerToken_get xs 4294967296)
 return 0
}
shadow main {assert true}
''', True)

    def test_native_declared_and_callback_precedence(self):
        self.native_routes('native-declared-list', '''fn list_Item_new()->int{return 17}
shadow list_Item_new {assert (== (list_Item_new) 17)}
fn ordinary()->int{return 23}
shadow ordinary {assert (== (ordinary) 23)}
fn invoke(list_Other_new:fn()->int)->int{return (list_Other_new)}
shadow invoke {assert (== (invoke ordinary) 23)}
fn main()->int{assert (== (list_Item_new) 17) assert (== (invoke ordinary) 23) return 0}
shadow main {assert (== (main) 0)}
''')

    def test_native_foreign_declaration_and_provider_collision(self):
        foreign = self.work / 'foreign-list.c'
        foreign.write_text('#include <stdint.h>\nint64_t list_Foreign_new(void) { return 31; }\n')
        provider = self.work / 'foreign-list.o'
        self.command('foreign-provider-build', [*self.native_cc, *self.native_flags, '-std=c99',
            '-Wall', '-Wextra', '-Werror', '-c', foreign, '-o', provider])
        self.native_routes('native-foreign-list', '''extern fn list_Foreign_new()->int
fn main()->int { let mut value:int=0 unsafe { set value (list_Foreign_new) } assert (== value 31) return 0 }
shadow main { assert true }
''', extra_links=(provider,))
        source = self.work / 'provider-collision.nano'
        source.write_text('''struct Item {value:int}
fn list_Item_get(value:int)->int{return value}
shadow list_Item_get {assert (== (list_Item_get 9) 9)}
fn main()->int{let xs:List<Item> =(list_Item_new) (list_Item_free xs) return 0}
shadow main {assert true}
''')
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            output = self.work / ('provider-collision-' + compiler)
            output.write_bytes(retained.SENTINEL)
            out, err = self.command('provider-collision-' + compiler,
                [ROOT / 'bin' / compiler, source, '-o', output, '--keep-c'],
                expected=tuple(range(1, 126)), timeout=300)
            self.assertEqual(output.read_bytes(), retained.SENTINEL)
            self.assertIn(b'I cannot share a native list provider symbol with a declaration.', out + err)
            self.assertNotIn(b'C compilation failed', out + err)

    def test_native_full_width_refusals(self):
        for element, value in (('int', '7'), ('string', '"seven"'), ('Item', 'Item{value:7}')):
            for position, index in enumerate(('4294967296', '9223372036854775807', '(- 0 9223372036854775807)')):
                source = f'''struct Item {{value:int}}
fn main()->int{{let xs:List<{element}> =(list_{element}_new) (list_{element}_push xs {value}) (list_{element}_get xs {index}) return 0}}
shadow main {{assert true}}
'''
                # These corrected guards terminate before indexing. They are not
                # old unchecked-provider reproductions or NanoISA admissions.
                self.native_routes('native-index-' + element + '-' + str(position), source, True)

        for element in ('int', 'string'):
            self.native_routes('native-capacity-' + element, f'''fn main()->int{{(list_{element}_with_capacity 4294967296) return 0}}
shadow main {{assert true}}
''', True)


if __name__ == '__main__':
    unittest.main()
