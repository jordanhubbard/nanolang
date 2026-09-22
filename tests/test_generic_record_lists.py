"""I retain nominal-list and evaluator lifetime controls before assertions.

My caller prepares fresh compilers/providers. I do not replace bootstrap with
this fixture, suppress sanitizer findings, or relabel legacy allocation as checked.
"""
from pathlib import Path
import hashlib
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/evaluator_lists'
SENTINEL = b'I retain the previous output.\n'


class GenericRecordLists(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='nano-record-lists-', dir=os.environ.get('NANO_LIST_REPORT_DIR')))
        print('I retain record-list artifacts at', cls.work, flush=True)
        cls.sequence = 0
        (cls.work / 'temporary').mkdir()
        cls.cc = shlex.split(os.environ.get('NANO_LIST_CC', 'cc'))
        cls.flags = shlex.split(os.environ.get('NANO_LIST_CFLAGS', ''))
        cls.links = shlex.split(os.environ.get('NANO_LIST_LDFLAGS', ''))
        cls.objects = [Path(x) for x in shlex.split(os.environ['NANO_LIST_OBJECTS'])]
        cls.objects = [p if p.is_absolute() else ROOT / p for p in cls.objects]
        if not cls.objects or any(not p.is_file() for p in cls.objects):
            raise AssertionError('I require all explicit prepared compiler providers.')
        cls.input_map = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.objects}
        (cls.work / 'provider-inputs-before.json').write_text(json.dumps(cls.input_map, indent=2) + '\n')
        (cls.work / 'selected-config.json').write_text(json.dumps(dict(cc=cls.cc, flags=cls.flags,
            links=cls.links, objects=list(map(str, cls.objects))), indent=2) + '\n')
        cls.owned = cls.work / 'owned-lifetimes'
        cls.programs = cls.work / 'parsed-lifetimes'
        cls.coroutine_errors = cls.work / 'coroutine-errors'
        cls.array_identity = cls.work / 'array-identity'
        cls.array_allocations = cls.work / 'array-allocations'
        common = [*cls.cc, *cls.flags, '-std=c99', '-Wall', '-Wextra', '-Werror', '-I', ROOT / 'src']
        cls.command('coroutine-error-build', [*common, ROOT / 'tests/test_coroutine_error_allocation.c',
            *cls.links, '-o', cls.coroutine_errors])
        cls.command('array-identity-build', [*common, ROOT / 'tests/test_nominal_array_identity.c',
            ROOT / 'tests/test_nominal_constructor_allocations.c',
            *[p for p in cls.objects if p.name != 'typechecker.o'], *cls.links, '-o', cls.array_identity])
        cls.command('array-allocation-build', [*common, ROOT / 'tests/test_nominal_array_allocations.c',
            ROOT / 'tests/test_nominal_array_allocations_checker.c',
            *[p for p in cls.objects if p.name not in ('env.o', 'typechecker.o')], *cls.links, '-o', cls.array_allocations])
        cls.command('owned-build', [*common, '-DEVALUATOR_ALLOCATION_HOOKS', ROOT / 'tests/test_evaluator_owned_lifetimes.c',
            ROOT / 'tests/test_evaluator_lifetime_eval.c', ROOT / 'tests/test_evaluator_lifetime_module.c',
            *[p for p in cls.objects if p.name not in ('env.o', 'eval.o', 'module.o')], *cls.links, '-o', cls.owned])
        cls.command('parsed-build', [*common, ROOT / 'tests/test_evaluator_lifetime_programs.c',
            ROOT / 'tests/test_evaluator_lifetime_eval.c', ROOT / 'tests/test_evaluator_lifetime_module.c',
            *[p for p in cls.objects if p.name not in ('eval.o', 'module.o')], *cls.links, '-o', cls.programs])

    @classmethod
    def tearDownClass(cls):
        after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.objects}
        (cls.work / 'provider-inputs-after.json').write_text(json.dumps(after, indent=2) + '\n')
        if after != cls.input_map:
            raise AssertionError('My prepared provider bytes changed during the fixture.')

    @classmethod
    def products(cls, prefix):
        store = cls.work / 'objects'; store.mkdir(exist_ok=True)
        rows = {}
        for path in sorted(cls.work.rglob('*')):
            if not path.is_file() or store in path.parents or path.suffix in ('.json', '.stdout', '.stderr'):
                continue
            data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
            target = store / digest
            if not target.exists(): target.write_bytes(data)
            rows[str(path.relative_to(cls.work))] = dict(sha256=digest, bytes=len(data))
        (cls.work / (prefix + '-products.json')).write_text(json.dumps(rows, indent=2) + '\n')

    @classmethod
    def command(cls, name, args, timeout=180, expected=(0,), extra=None):
        cls.sequence += 1; name = f'{cls.sequence:04d}-{name}'
        args = list(map(str, args))
        env = dict(os.environ, TMPDIR=str(cls.work / 'temporary'), ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',
                   UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', LSAN_OPTIONS='')
        if extra: env.update(extra)
        cls.products(name + '-before')
        (cls.work / (name + '-command.json')).write_text(json.dumps(dict(argv=args, cwd=str(ROOT),
            timeout=timeout, expected=list(expected), extra=extra, TMPDIR=env['TMPDIR'], ASAN_OPTIONS=env['ASAN_OPTIONS'],
            UBSAN_OPTIONS=env['UBSAN_OPTIONS'], LSAN_OPTIONS=env['LSAN_OPTIONS']), indent=2) + '\n')
        status = dict(returncode=None, timeout=False, cleanup_errors=[], group_absent=False)
        start = time.monotonic(); process = None
        outpath, errpath = cls.work / (name + '.stdout'), cls.work / (name + '.stderr')
        with outpath.open('wb') as stdout, errpath.open('wb') as stderr:
            try:
                process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                try: status['returncode'] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    status['timeout'] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: pass
            except Exception as error:
                status['launch_or_wait_error'] = repr(error)
            finally:
                if process is not None:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    except OSError as error: status['cleanup_errors'].append(repr(error))
                    try: status['child_returncode'] = process.wait(timeout=5)
                    except subprocess.TimeoutExpired: status['cleanup_errors'].append('I exhausted my bounded post-kill wait.')
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        try: os.killpg(process.pid, 0)
                        except ProcessLookupError: status['group_absent'] = True; break
                        except OSError as error: status['cleanup_errors'].append(repr(error)); break
                        time.sleep(0.02)
                    if not status['group_absent']: status['cleanup_errors'].append('I still observe the process group.')
                status['elapsed'] = time.monotonic() - start
                stdout.flush(); stderr.flush()
                (cls.work / (name + '-status.json')).write_text(json.dumps(status, indent=2) + '\n')
                cls.products(name + '-after')
        out, err = outpath.read_bytes(), errpath.read_bytes()
        if (status.get('launch_or_wait_error') or status['timeout'] or status['returncode'] not in expected
                or status['cleanup_errors'] or not status['group_absent']):
            raise AssertionError((name, status, out[-8000:], err[-8000:]))
        if b'AddressSanitizer' in err or b'LeakSanitizer' in err or b'runtime error:' in err:
            raise AssertionError((name, err))
        return out, err

    def test_checked_storage_scheduler_and_allocation_prefixes(self):
        self.command('task-context', [self.programs, 'task-context'])
        for mode in ('task-foreign-ready', 'task-foreign-done', 'task-raw'):
            out, err = self.command(mode, [self.programs, mode], expected=(1,))
            self.assertEqual(err, 'I cannot access a task owned by another Environment.\n')
        self.command('coroutine-error-allocations', [self.coroutine_errors])
        out, _ = self.command('checked-lifetimes', [self.owned])
        self.assertIn(b'checked ownership assertions', out)
        self.command('checked-callable-hook', [self.programs, 'callable'])

    def test_parsed_evaluator_and_public_escape(self):
        for name in ('mutations', 'staging', 'imported', 'escape', 'async'):
            self.command('evaluator-' + name, [self.programs, 'escape' if name == 'escape' else 'program', FIXTURES / (name + '.nano')])

    def test_array_source_paths_and_producers(self):
        for name in ('array_paths', 'array_producers'):
            source = (FIXTURES / (name + '.nano')).read_text()
            source = source.replace('"array_records.nano"', '"' + str(FIXTURES / 'array_records.nano') + '"')
            self.source_routes(name, source)

    def test_record_field_destinations(self):
        prelude = '''struct Item { value: int }
struct Pair { left: int, right: int }
struct Strings { values: array<string> }
struct Holder { item: Item, values: array<Item>, numbers: array<array<int>> }
'''
        positive = prelude + '''fn main() -> int {
 let strings: Strings = Strings { values: (array_new 1 "ready") }
 assert (== (at strings.values 0) "ready")
 let source: array<Item> = [Item { value: 9 }]
 let made: array<Item> = (array_new 1 (Item { value: 11 }))
 assert (== (at made 0).value 11)
 let value: Holder = Holder { numbers: [[1], []], values: source, item: Item { value: 7 } }
 assert (== value.item.value 7)
 assert (== (at value.values 0).value 9)
 assert (== (array_length (at value.numbers 1)) 0)
 let empty: Holder = Holder { item: Item { value: 0 }, values: [], numbers: [] }
 assert (== (array_length empty.values) 0)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        self.source_routes('record-fields-positive', positive)
        cases = {
            'scalar': 'let item = Item { value: true }',
            'missing': 'let pair = Pair { left: 1 }',
            'duplicate': 'let pair = Pair { left: 1, left: 2 }',
            'unknown': 'let item = Item { other: 1 }',
            'mixed-array': 'let value = Holder { item: Item { value: 1 }, values: [Item { value: 2 }, 3], numbers: [] }',
            'nested-array': 'let value = Holder { item: Item { value: 1 }, values: [], numbers: [[1], [true]] }',
            'nested-record': 'let value = Holder { item: Pair { left: 1, right: 2 }, values: [], numbers: [] }',
        }
        for name, body in cases.items():
            self.source_routes('record-field-refuse-' + name,
                prelude + 'fn main() -> int { ' + body + ' return 0 }\nshadow main { assert true }\n',
                reject=True, checker_refusal=True)

    def test_discarded_array_value_facts(self):
        positive = '''fn main() -> int {
 []
 [[], []]
 if true { 1 } else { false }
 if false { (print "") }
 (cond (true []) (else [[1]]))
 match 0 { 0 => [] _ => [1] }
 return 0
}
shadow main { assert (== (main) 0) }
'''
        self.source_routes('discarded-array-positive', positive)
        self.source_routes('block-expression-positive', '''fn main() -> int {
 let rows: array<array<int>> = (cond (true { let local: int = 7 [[local]] }) (else []))
 let value: int = ({ let outer: int = 2 { let inner: int = 5 (+ outer inner) } })
 { let value: bool = true assert value }
 assert (== (at (at rows 0) 0) value)
 return 0
}
shadow main { assert (== (main) 0) }
''')
        self.source_routes('if-expression-positive', '''fn choose(flag: bool) -> int {
 return if flag { 7 } else { 9 }
}
shadow choose { assert (== (choose true) 7) assert (== (choose false) 9) }
fn main() -> int {
 let rows: array<array<int>> = (if false { [] } else if true { [[7]] } else { [[9]] })
 let value: int = (if true { (if false { 1 } else { 7 }) } else { 9 })
 if true { 1 } else { false }
 if false { (print "") }
 assert (== (at (at rows 0) 0) value)
 assert (== value (choose true))
 return 0
}
shadow main { assert (== (main) 0) }
''')
        for name, body in {
            'block-local-escape': '{ let local: int = 7 } return local',
            'block-destination': 'let values: array<int> = { [false] } return 0',
            'if-discarded': '(if true { [1] } else { [false] }) return 0',
            'if-condition': 'let value: int = (if 1 { 7 } else { 9 }) return 0',
            'if-nested': 'let value: array<int> = (if true { [] } else if false { [1] } else { [false] }) return 0',
            'literal': '[1, true] return 0',
            'unreachable-literal': 'return 0 [1, true]',
            'cond': '(cond (true [1]) (else [false])) return 0',
            'match': 'match 0 { 0 => [1] _ => [false] } return 0',
            'nested-block': '{ [1, true] } return 0',
        }.items():
            self.source_routes('discarded-array-refuse-' + name,
                'fn main() -> int { ' + body + ' }\nshadow main { assert true }\n',
                True, checker_refusal=True)

    def test_selfhost_array_value_boundaries(self):
        prelude = '''struct Item { value: int }
struct Other { value: int }
struct Rows { values: array<array<Item>> }
union Wrapped { Rows { values: array<array<Item>> } }
fn make(n: int) -> Item { return Item { value: n } }
shadow make { assert (== (make 7).value 7) }
fn total(rows: array<array<Item>>) -> int {
 let mut sum: int = 0
 for row in rows { for item in row { set sum (+ sum item.value) } }
 return sum
}
shadow total { assert (== (total [[], [(make 7)]]) 7) }
fn choose(flag: bool) -> array<array<Item>> {
 return (cond (flag { let row: array<Item> = [(make 7)] [[], row] }) (else [[]]))
}
shadow choose { assert (== (total (choose true)) 7) assert (== (total (choose false)) 0) }
fn selected(n: int) -> array<Item> { return (match n { 0 => [] _ => [(make 9)] }) }
shadow selected { assert (== (array_length (selected 0)) 0) assert (== (at (selected 1) 0).value 9) }
fn terminal(flag: bool) -> array<Item> {
 return (cond (flag { return [(make 11)] }) (else []))
}
shadow terminal { assert (== (at (terminal true) 0).value 11) assert (== (array_length (terminal false)) 0) }
fn wrong_predicate(value: Item) -> int { return value.value }
shadow wrong_predicate { assert (== (wrong_predicate (make 5)) 5) }
'''
        positive = prelude + '''fn main() -> int {
 let inferred = [[], [(make 3)], []]
 assert (== (total inferred) 3)
 let direct: array<array<Item>> = [[], [(make 4)]]
 let held: Rows = Rows { values: [[], [(make 5)]] }
 assert (== (total held.values) 5)
 let call: fn(array<array<Item>>) -> int = total
 assert (== (call [[], [(make 6)]]) 6)
 assert (== (total (choose true)) 7)
 assert (== (array_length (selected 0)) 0)
 assert (== (at (terminal true) 0).value 11)
 let sliced = (array_slice direct 0 2)
 assert (== (total sliced) 4)
 let added = (array_push [[], [(make 1)]] [(make 2)])
 assert (== (total added) 3)
 (array_set added 0 [(make 8)])
 assert (== (total added) 11)
 let wrapped: Wrapped = Wrapped.Rows { values: [[], [(make 12)]] }
 match wrapped { Rows(payload) => { assert (== (total payload.values) 12) } }
 return 0
}
shadow main { assert (== (main) 0) }
'''
        self.source_routes('array-value-boundaries-positive', positive)
        wrong = {
            'typed-later-member': 'let values: array<array<Item>> = [[], [Other { value: 1 }]]',
            'inferred-later-member': 'let values = [[], [(make 1)], [Other { value: 2 }]]',
            'heterogeneous': 'let values = [[], [1], [true]]',
            'direct-literal': '(total [[], [Other { value: 1 }]])',
            'indirect-literal': 'let call: fn(array<array<Item>>) -> int = total (call [[], [Other { value: 1 }]])',
            'union-payload': 'let value: Wrapped = Wrapped.Rows { values: [[], [Other { value: 1 }]] }',
            'if-local': 'let values: array<array<Item>> = (if true { let row: array<Item> = [(make 1)] [row] } else { let row: array<Other> = [Other { value: 2 }] [row] })',
            'match-local': 'let values: array<array<Item>> = (match 0 { 0 => [[]] _ => { let row: array<Other> = [Other { value: 2 }] [row] } })',
            'nested-iteration': 'let rows: array<array<Other>> = [[Other { value: 2 }]] for row in rows { let values: array<Item> = row }',
            'ignored-nested-push': 'let rows: array<array<Item>> = [[(make 1)]] (array_push rows [Other { value: 2 }])',
            'ignored-nested-set': 'let rows: array<array<Item>> = [[(make 1)]] (array_set rows 0 [Other { value: 2 }])',
            'filter-result': 'let values: array<Item> = (filter [(make 1)] wrong_predicate)',
        }
        for name, body in wrong.items():
            self.source_routes('array-value-refuse-' + name,
                prelude + 'fn main() -> int { ' + body + ' return 0 }\nshadow main { assert true }\n',
                True, checker_refusal=True)
        self.source_routes('array-value-refuse-return', prelude + '''fn bad() -> array<array<Item>> {
 return [[], [Other { value: 2 }]]
}
shadow bad { assert true }
fn main() -> int { return 0 }
shadow main { assert true }
''', True, checker_refusal=True)

    def test_array_destination_and_origin_refusals(self):
        prelude = f'module "{FIXTURES / "array_records.nano"}" as records\n'
        prelude += '''struct Item { value: int }
struct Holder { values: array<Item> }
fn make() -> Item { return Item { value: 31 } }
shadow make { assert (== (make).value 31) }
fn consume(values: array<Item>) -> int { return (array_length values) }
shadow consume { assert (== (consume [(make)]) 1) }
fn keep(value: Item) -> bool { return true }
shadow keep { assert (keep (make)) }
'''
        wrong = {
            'initializer': 'let values: array<Item> = (records.values)',
            'assignment': 'let mut values: array<Item> = [] set values (records.values)',
            'field': 'let holder: Holder = Holder { values: (records.values) }',
            'qualified-field': 'let holder: records.Holder = records.Holder { values: [(make)] }',
            'direct': '(consume (records.values))',
            'qualified': '(records.forward [(make)])',
            'indirect': 'let f: fn(array<Item>) -> int = consume (f (records.values))',
            'inferred': 'let values = (records.values) let wrong: array<Item> = values',
            'nested': 'let values: array<array<Item>> = [[], (records.values)]',
            'mixed-literal': 'let values = [(make), (records.make 7)]',
            'index': 'let value: Item = (at (records.values) 0)',
            'iteration': 'for value in (records.values) { let wrong: Item = value }',
            'new': 'let values: array<Item> = (array_new 2 (records.make 7))',
            'push': 'let values: array<Item> = (array_push [(make)] (records.make 7))',
            'push-ignored': 'let values: array<Item> = [(make)] (array_push values (records.make 7))',
            'set-element': 'let values: array<Item> = [(make)] (array_set values 0 (records.make 7))',
            'slice': 'let values: array<Item> = (array_slice (records.values) 0 1)',
            'map-input': 'let values = (map (records.values) keep)',
            'filter-input': 'let values = (filter (records.values) keep)',
            'map-output': 'let values: array<Item> = (map (records.values) records.raise)',
            'if': 'let values: array<Item> = (if true { [(make)] } else { (records.values) })',
            'cond': 'let values: array<Item> = (cond (true [(make)]) (else (records.values)))',
            'match': 'let values: array<Item> = (match true { true => { [(make)] } false => { (records.values) } })',
            'empty-unresolved': 'let values: array<Missing> = []',
        }
        for name, body in wrong.items():
            self.source_routes('array-refuse-' + name, prelude + 'fn main() -> int { ' + body + ' return 0 }\nshadow main { assert true }\n', True)
        self.source_routes('array-refuse-return', prelude + '''fn wrong() -> array<Item> { return (records.values) }
shadow wrong { assert true }
fn main() -> int { return 0 }
shadow main { assert true }
''', True)
        # I exercise the second module-checking pass with the same wrong global boundary.
        module = self.work / 'array-wrong-module.nano'
        module.write_text(prelude + 'let wrong: array<Item> = (records.values)\n')
        self.source_routes('array-refuse-module-global', f'module "{module}" as bad\nfn main() -> int {{ return 0 }}\nshadow main {{ assert true }}\n', True, import_refusal=True)

    def test_array_union_definition_and_substitution_owners(self):
        prelude = f'module "{FIXTURES / "array_records.nano"}" as records\n'
        prelude += '''struct Item { value: int }
fn make() -> Item { return Item { value: 31 } }
shadow make { assert (== (make).value 31) }
'''
        both = 'records.Mixed<Item>.Both { fixed: (records.values), supplied: [(make)] }'
        source = prelude + '''fn main() -> int {
 let value: records.Mixed<Item> = BOTH
 let nested: records.Nested<Item> = records.Nested<Item>.Wrapped { inner: BOTH }
 let empty: records.Mixed<Item> = records.Mixed<Item>.Both { fixed: [], supplied: [] }
 match value { Both(payload) => {
  assert (== (at payload.fixed 0).value 7)
  assert (== (at payload.supplied 0).value 31)
 } }
 match nested { Wrapped(outer) => { match outer.inner { Both(payload) => {
  assert (== (at payload.fixed 1).value 9)
  assert (== (at payload.supplied 0).value 31)
 } } } }
 match empty { Both(payload) => { assert (== (array_length payload.fixed) 0) assert (== (array_length payload.supplied) 0) } }
 return 0
}
shadow main { assert (== (main) 0) }
'''
        self.source_routes('array-union-mixed-owners', source.replace('BOTH', both))
        for name, expression in (
            ('fixed', both.replace('fixed: (records.values)', 'fixed: [(make)]')),
            ('substituted', both.replace('supplied: [(make)]', 'supplied: (records.values)'))):
            for nested in (False, True):
                annotation = 'records.Nested<Item>' if nested else 'records.Mixed<Item>'
                value = 'records.Nested<Item>.Wrapped { inner: ' + expression + ' }' if nested else expression
                self.source_routes('array-union-refuse-' + name + ('-nested' if nested else ''),
                    prelude + 'fn main() -> int { let value: ' + annotation + ' = ' + value + ' return 0 }\nshadow main { assert true }\n', True)

    def test_array_intrinsic_and_declaration_identity(self):
        self.command('array-identity', [self.array_identity])
        self.command('array-allocations', [self.array_allocations])

    def test_deferred_foreign_declarations(self):
        self.command('foreign-identity-facts', [self.programs, 'foreign-facts'])
        directory = self.work / 'foreign-declarations'; directory.mkdir()
        contracts = directory / 'contracts.nano'
        contracts.write_text('module Contracts\nextern struct NativeHolder { tokens: List<NativeToken> }\n')
        tokens = directory / 'tokens.nano'
        tokens.write_text('module Tokens\npub extern struct NativeToken { value: int }\n')
        self.command('foreign-pending', [self.programs, 'declarations-pending', contracts])
        # The importer supplies the extern only after the imported pending field.
        importer = directory / 'importer.nano'
        importer.write_text(f'module Tokens\nimport "{contracts}"\nextern struct NativeToken {{ value: int }}\n')
        self.command('foreign-importer-forward', [self.programs, 'declarations-resolved', importer])
        for order, modules in enumerate(((contracts, tokens), (tokens, contracts))):
            source = directory / f'order-{order}.nano'
            source.write_text(''.join(f'import "{module}"\n' for module in modules))
            self.command(f'foreign-order-{order}', [self.programs, 'declarations-resolved', source])
        aliased = directory / 'aliased.nano'
        aliased.write_text(f'import "{tokens}" as native\nextern struct NativeHolder {{ tokens: List<native.NativeToken> }}\n')
        self.command('foreign-alias', [self.programs, 'declarations-alias', aliased])
        # I do not turn an unrelated ordinary spelling into foreign authority.
        ordinary = directory / 'ordinary.nano'
        ordinary.write_text('module Ordinary\npub struct NativeToken { value: int }\n')
        unrelated = directory / 'unrelated.nano'
        unrelated.write_text(f'import "{contracts}"\nimport "{ordinary}"\n')
        self.command('ordinary-not-foreign', [self.programs, 'declarations-pending', unrelated])
        for other in (ordinary, directory / 'competing.nano'):
            if other != ordinary:
                other.write_text('module Competing\nextern struct NativeToken { value: int }\n')
            for order, modules in enumerate(((tokens, other), (other, tokens))):
                source = directory / f'collision-{other.stem}-{order}.nano'
                source.write_text(''.join(f'import "{module}"\n' for module in modules))
                _, err = self.command('foreign-collision', [self.programs, 'declarations-import-refuse', source])
                self.assertIn(b'colliding foreign record declarations', err)
        unresolved = directory / 'use.nano'
        unresolved.write_text(f'import "{contracts}"\nfn main() -> int {{ let xs: List<NativeToken> = (list_NativeToken_new) return 0 }}\nshadow main {{ assert true }}\n')
        _, err = self.command('pending-value-refused', [self.programs, 'declarations-refuse', unresolved])
        self.assertIn(b"Unknown type 'NativeToken'", err)
        self.assertNotIn(b'specialization collision', err)
        ordinary_missing = directory / 'ordinary-missing.nano'
        ordinary_missing.write_text('struct Holder { values: List<MissingRecord> }\n')
        _, err = self.command('ordinary-field-unresolved', [self.programs, 'declarations-refuse', ordinary_missing])
        self.assertIn(b"cannot resolve this list's exact record declaration", err)
        self.assertNotIn(b'specialization collision', err)
        # Ordinary same-module forward collection also needs both checker routes.
        forward = 'struct Holder { values: List<Element> }\nstruct Element { value: int }\nfn main() -> int { let xs: List<Element> = (list_Element_new) return 0 }\nshadow main { assert (== (main) 0) }\n'
        self.source_routes('ordinary-forward-list', forward)

    def test_unqualified_nominal_import_provenance(self):
        directory = self.work / 'nominal-imports'; directory.mkdir()
        origin = directory / 'origin.nano'
        origin.write_text('''module ImportOrigin
pub struct Token { value: int }
pub struct Packet { items: array<Token> }
pub fn import_token() -> Token { return Token { value: 17 } }
shadow import_token { assert (== (import_token).value 17) }
pub fn import_packet() -> Packet { return Packet { items: [(import_token)] } }
shadow import_packet { assert (== (at (import_packet).items 0).value 17) }
''')
        wrapper = directory / 'wrapper.nano'
        wrapper.write_text(f'from "{origin}" import Token, Packet, import_packet\n' + '''module ImportWrapper
pub fn imported_fields(value: Packet) -> array<Token> { return value.items }
shadow imported_fields { assert (== (at (imported_fields (import_packet)) 0).value 17) }
''')
        body = '''fn main() -> int {
 let packet: Packet = (import_packet)
 let items: array<Token> = (imported_fields packet)
 let copied: array<Token> = items
 assert (== (at copied 0).value 17)
 let values: List<Token> = (list_Token_new)
 (list_Token_push values (at items 0))
 assert (== (list_Token_get values 0).value 17)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        for name, declaration in (('selective', f'from "{origin}" import Token, Packet, import_packet'),
                                  ('plain', f'import "{origin}"'),
                                  ('wildcard', f'from "{origin}" import *')):
            self.source_routes('nominal-' + name, declaration + f'\nfrom "{wrapper}" import imported_fields\n' + body)
        # Definition-site leaves remain usable without leaking a wrapper's aliases.
        inferred = f'from "{origin}" import import_packet\nfrom "{wrapper}" import imported_fields\n' + '''fn main() -> int {
 let result = (imported_fields (import_packet))
 assert (== (at result 0).value 17)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        self.source_routes('nominal-definition-site', inferred)
        local = f'import "{origin}"\n' + '''struct Token { text: string }
fn main() -> int {
 let values: array<Token> = [Token { text: "local" }]
 assert (== (at values 0).text "local")
 return 0
}
shadow main { assert (== (main) 0) }
'''
        self.source_routes('nominal-local-precedence', local)
        other = directory / 'other.nano'
        other.write_text('module OtherOrigin\npub struct Token { value: int }\n')
        for order, modules in enumerate(((origin, other), (other, origin))):
            source = ''.join(f'import "{module}"\n' for module in modules) + 'fn main() -> int { return 0 }\nshadow main { assert true }\n'
            self.source_routes('nominal-conflict-' + str(order), source, True, import_refusal=True)
        cross = directory / 'cross-kind.nano'
        cross.write_text('module CrossKind\npub union Token { One { value: int } }\n')
        enum = directory / 'enum-kind.nano'
        enum.write_text('module EnumKind\npub enum Token { One, Two }\n')
        for first, second in ((origin, cross), (cross, origin), (origin, enum), (enum, origin), (cross, enum), (enum, cross)):
            source = f'import "{first}"\nimport "{second}"\nfn main() -> int {{ return 0 }}\nshadow main {{ assert true }}\n'
            self.source_routes('nominal-kinds-' + first.stem + '-' + second.stem, source, True, import_refusal=True)
        self.source_routes('nominal-local-record-imported-union', f'import "{cross}"\n' + local.split('\n', 1)[1])
        self.source_routes('nominal-repeat', f'import "{origin}"\nimport "{origin}"\nfrom "{wrapper}" import imported_fields\n' + body)
        missing = f'from "{origin}" import Missing\nfn main() -> int {{ let values: array<Missing> = [] return 0 }}\nshadow main {{ assert true }}\n'
        self.source_routes('nominal-undefined', missing, True)
        unbound = f'from "{wrapper}" import imported_fields\nfn main() -> int {{ let values: array<Token> = [] return 0 }}\nshadow main {{ assert true }}\n'
        self.source_routes('nominal-no-namespace-promotion', unbound, True)
        renamed = f'from "{origin}" import Token as Renamed\nfn main() -> int {{ return 0 }}\nshadow main {{ assert true }}\n'
        self.source_routes('nominal-type-alias-still-refused', renamed, True, import_refusal=True)

    def test_real_cache_generations_and_private_compiler(self):
        # I use the actual private compiler, including a fresh missing-input path.
        module = self.work / 'cache-input.nano'
        module.write_text('module LeaseFixture\npub fn answer() -> int { return 7 }\nshadow answer { assert (== (answer) 7) }\n')
        self.command('cache-registration-prefix', [self.owned, 'cache-registration', module])
        self.command('cache-init-baseline', [self.owned, 'cache-init', 'all', '0'])
        for once in (0, 1):
            for prefix in range(4):
                out, err = self.command(f'cache-init-{once}-{prefix}', [self.owned, 'cache-init', prefix, once], expected=(1,))
                self.assertIn(b'I released every observed cache initialization allocation.', out)
                self.assertIn(b'I cannot allocate a module cache', err)
                self.command(f'cache-recovery-{once}-{prefix}', [self.owned, 'cache-init', 'all', '0'])
        self.command('cache-private-compile', [self.programs, 'cache', module, self.work / 'cache-output.o'])
        for mode, diagnostic in (
            ('cache-refuse', b'I cannot clear a module cache with pending evaluator tasks.'),
            ('env-refuse', b'I cannot destroy an Environment with pending evaluator tasks.')):
            _, err = self.command(mode, [self.programs, mode, module, self.work / (mode + '.o')], expected=(1,))
            self.assertIn(diagnostic, err)

    def source_routes(self, name, source, reject=False, vm=True, import_refusal=False, checker_refusal=False):
        path = self.work / (name + '.nano'); path.write_text(source)
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            output = self.work / (name + '-' + compiler); output.write_bytes(SENTINEL)
            out, err = self.command(name + '-' + compiler + '-build',
                [ROOT / 'bin' / compiler, path, '-o', output, '--keep-c'], expected=tuple(range(1, 126)) if reject else (0,), timeout=300)
            if reject:
                self.assertEqual(output.read_bytes(), SENTINEL)
                self.assertNotIn(b'C compilation failed', out + err)
                if checker_refusal:
                    if compiler == 'nanoc_c':
                        self.assertTrue(any(token in out + err for token in
                            (b'E001 TYPE MISMATCH', b'E003 ARITY MISMATCH', b'E004 UNKNOWN FIELD',
                             b'nominal', b'Nominal', b'array element')), (out, err))
                    else:
                        self.assertIn(b'NSType checking failed', out + err)
            else: self.command(name + '-' + compiler + '-run', [output], timeout=15)
        # I exercise checker refusal in the actual evaluator too, before run_program.
        mode = 'declarations-import-refuse' if import_refusal else 'reject' if reject else 'program'
        self.command(name + '-evaluator', [self.programs, mode, path])
        if vm:
            output = self.work / (name + '.nvm'); output.write_bytes(SENTINEL)
            self.command(name + '-emit', [ROOT / 'bin/nano_virt', path, '--emit-nvm', '-o', output],
                         expected=tuple(range(1, 126)) if reject else (0,), timeout=300)
            if reject: self.assertEqual(output.read_bytes(), SENTINEL)
            else: self.command(name + '-vm', [ROOT / 'bin/nano_vm', output], timeout=30)

    def test_nominal_boundaries_all_source_producers(self):
        prelude = '''struct A { value: int }
struct B { value: int }
struct Held { value: A, values: List<A> }
fn make() -> A { return A { value: 7 } }
shadow make { assert (== (make).value 7) }
fn take(value: A) -> int { return value.value }
shadow take { assert (== (take (make)) 7) }
fn lists(values: List<A>) -> int { return (list_A_length values) }
shadow lists { assert (== (lists (list_A_new)) 0) }
'''
        wrong = {
            'record-init': 'let value: A = B { value: 8 }',
            'record-set': 'let mut value: A = (make) set value B { value: 8 }',
            'record-call': '(take B { value: 8 })',
            'record-indirect': 'let f: fn(A) -> int = take (f B { value: 8 })',
            'record-field': 'let h: Held = Held { value: B { value: 8 }, values: (list_A_new) }',
            'list-init': 'let xs: List<A> = (list_B_new)',
            'list-set': 'let mut xs: List<A> = (list_A_new) set xs (list_B_new)',
            'list-field': 'let h: Held = Held { value: (make), values: (list_B_new) }',
            'list-call': '(lists (list_B_new))',
            'list-indirect': 'let f: fn(List<A>) -> int = lists (f (list_B_new))',
            'list-integer': 'let xs: List<A> = 1',
            'push': 'let xs: List<A> = (list_A_new) (list_A_push xs B { value: 8 })',
            'insert': 'let xs: List<A> = (list_A_new) (list_A_insert xs 0 B { value: 8 })',
            'set': 'let xs: List<A> = (list_A_new) (list_A_set xs 0 B { value: 8 })',
            'get-wrong': 'let xs: List<B> = (list_B_new) let value: A = (list_B_get xs 0)',
            'callback-result': 'let f: fn() -> B = make',
            'callback-argument': 'let f: fn(B) -> int = take',
        }
        for name, body in wrong.items():
            self.source_routes('refuse-' + name, prelude + 'fn main() -> int { ' + body + ' return 0 }\nshadow main { assert true }\n', True)
        for name, result, expression in [('record-return', 'A', 'B { value: 8 }'), ('list-return', 'List<A>', '(list_B_new)')]:
            self.source_routes('refuse-' + name, prelude + f'fn wrong() -> {result} {{ return {expression} }}\nshadow wrong {{ assert true }}\nfn main() -> int {{ return 0 }}\nshadow main {{ assert true }}\n', True)
        positive = prelude + '''fn forward(xs: List<A>) -> List<A> { return xs }
shadow forward { assert (== (lists (forward (list_A_new))) 0) }
fn main() -> int { let mut xs: List<A> = (list_A_new) set xs (forward xs)
 (list_A_insert xs 0 (make)) let removed: A = (list_A_remove xs 0)
 assert (== (take removed) 7) return 0 }
shadow main { assert (== (main) 0) }
'''
        self.source_routes('all-boundaries-positive', positive)

    def test_nested_nominal_annotations_and_branch_agreement(self):
        prelude = f'module "{FIXTURES / "left.nano"}" as left\nmodule "{FIXTURES / "right.nano"}" as right\n'
        for name, wanted, actual in (
            ('array', 'array<left.Item>', 'array<right.Item>'),
            ('nested-array', 'array<array<left.Item>>', 'array<array<right.Item>>'),
            ('tuple', '(left.Item, int)', '(right.Item, int)'),
            ('nested-callable', 'fn(left.Item) -> int', 'fn(right.Item) -> int')):
            source = prelude + f'fn consume(value: {actual}) -> int {{ return 0 }}\nshadow consume {{ assert true }}\n'
            source += f'fn main() -> int {{ let f: fn({wanted}) -> int = consume return 0 }}\nshadow main {{ assert true }}\n'
            self.source_routes('nested-refuse-' + name, source, True)
        for kind, expression in (
            ('cond', '(cond (flag left.read) (else left.read))'),
            ('if', '(if flag { left.read } else { left.read })'),
            ('match', '(match flag { true => { left.read } false => { left.read } })')):
            source = prelude + f'fn choose(flag: bool) -> fn(left.Item) -> int {{ return {expression} }}\n'
            source += 'shadow choose { assert (== ((choose true) (left.make 7)) 7) assert (== ((choose false) (left.make 8)) 8) }\n'
            source += 'fn main() -> int { assert (== ((choose true) (left.make 7)) 7) return 0 }\nshadow main { assert (== (main) 0) }\n'
            self.source_routes('callable-branch-' + kind, source)
            wrong = source.replace('else left.read', 'else right.read').replace('else { left.read', 'else { right.read').replace('false => { left.read', 'false => { right.read')
            self.source_routes('callable-branch-refuse-' + kind, wrong, True)

    def test_imported_identity_and_callable_branches(self):
        prelude = f'import "{FIXTURES / "left.nano"}" as left\nimport "{FIXTURES / "right.nano"}" as right\n'
        for name, body in {
            'record': 'let value: left.Item = (right.make 8)',
            'list': 'let xs: List<left.Item> = (right.make_list)',
            'direct': '(left.read (right.make 8))',
            'indirect': 'let f: fn(left.Item) -> int = left.read (f (right.make 8))',
            'signature': 'let f: fn(left.Item) -> int = right.read',
        }.items():
            self.source_routes('imported-refuse-' + name, prelude + 'fn main() -> int { ' + body + ' return 0 }\nshadow main { assert true }\n', True)
        # I retain the complete imported success fixture at its own path for relative imports.
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            output = self.work / ('imported-' + compiler)
            self.command('imported-' + compiler, [ROOT / 'bin' / compiler, FIXTURES / 'imported.nano', '-o', output, '--keep-c'], timeout=300)
            self.command('imported-run-' + compiler, [output], timeout=15)

    def test_exact_list_and_trim_call_facts(self):
        # I retain actual supported native/evaluator library routes separately
        # from NanoISA admission. The original mutation/VM corpus is unchanged.
        source = """struct Item { value: int }
fn main() -> int {
 let xs: List<Item> = (list_Item_new)
 assert (list_Item_is_empty xs)
 (list_Item_push xs (Item { value: 7 }))
 (list_Item_set xs 0 (Item { value: 9 }))
 (list_Item_insert xs 1 (Item { value: 11 }))
 assert (== (list_Item_length xs) 2)
 assert (>= (list_Item_capacity xs) 2)
 let read: Item = (list_Item_get xs 0)
 let removed: Item = (list_Item_remove xs 0)
 let popped: Item = (list_Item_pop xs)
 assert (== read.value 9)
 assert (== removed.value 9)
 assert (== popped.value 11)
 (list_Item_push xs read)
 (list_Item_clear xs)
 assert (list_Item_is_empty xs)
 (list_Item_free xs)
 let words: array<string> = [(str_trim "  exact  ")]
 assert (== (at words 0) "exact")
 return 0
}
shadow main { assert (== (main) 0) }
"""
        self.source_routes('exact-list-and-trim', source, vm=False)
        for element, first, second in (('int', '7', '9'), ('string', '"seven"', '"nine"')):
            scalar = f"""fn main() -> int {{
 let xs: List<{element}> = (list_{element}_with_capacity 3)
 assert (list_{element}_is_empty xs)
 assert (>= (list_{element}_capacity xs) 3)
 (list_{element}_push xs {first})
 (list_{element}_insert xs 1 {second})
 (list_{element}_set xs 0 {second})
 assert (== (list_{element}_get xs 0) {second})
 assert (== (list_{element}_remove xs 0) {second})
 assert (== (list_{element}_pop xs) {second})
 (list_{element}_push xs {first})
 (list_{element}_clear xs)
 assert (== (list_{element}_length xs) 0)
 (list_{element}_free xs)
 return 0
}}
shadow main {{ assert (== (main) 0) }}
"""
            self.source_routes('exact-scalar-list-' + element, scalar, vm=False)
        prelude = 'struct Item { value: int }\nstruct Other { value: int }\n'
        for name, body in {
            'receiver': 'let xs: List<Other> = (list_Other_new) (list_Item_length xs)',
            'element': 'let xs: List<Item> = (list_Item_new) (list_Item_set xs 0 (Other { value: 1 }))',
            'index': 'let xs: List<Item> = (list_Item_new) (list_Item_get xs true)',
            'arity': '(list_Item_new 1)',
            'trim-type': '(str_trim 1)',
            'trim-arity': '(str_trim "x" "y")',
        }.items():
            self.source_routes('exact-list-refuse-' + name, prelude + 'fn main() -> int { ' + body + ' return 0 }\nshadow main { assert true }\n', True, vm=False, checker_refusal=True)

    def test_declared_call_precedence_and_explicit_enum_limit(self):
        source = """struct A { value: int }
fn list_A_insert(a: int, b: int, c: int) -> int { return (+ (+ a b) c) }
shadow list_A_insert { assert (== (list_A_insert 1 2 3) 6) }
fn List_A_new() -> int { return 19 }
shadow List_A_new { assert (== (List_A_new) 19) }
fn main() -> int {
 assert (== (list_A_insert 1 2 3) 6)
 assert (== (List_A_new) 19)
 let f: fn(int, int, int) -> int = list_A_insert
 assert (== (f 1 2 3) 6)
 return 0
}
shadow main { assert (== (main) 0) }
"""
        self.source_routes('declared-precedence', source)
        enum = 'enum Shade { Red, Blue }\nfn main() -> int { let xs: List<Shade> = (list_Shade_new) return 0 }\nshadow main { assert true }\n'
        self.source_routes('enum-list-still-refused', enum, True)
        global_wrong = 'struct A { value: int }\nstruct B { value: int }\nlet xs: List<A> = (list_B_new)\nfn main() -> int { return 0 }\nshadow main { assert true }\n'
        self.source_routes('global-list-refused', global_wrong, True)
        unresolved = 'fn use(value: array<Missing>) -> int { return 0 }\nshadow use { assert true }\nfn main() -> int { let f: fn(array<Missing>) -> int = use return 0 }\nshadow main { assert true }\n'
        self.source_routes('nested-unresolved-refused', unresolved, True)

    def test_mutation_order_all_routes_and_unchanged_lexer(self):
        self.source_routes('mutation-order', (FIXTURES / 'mutations.nano').read_text())
        # This is the exact previously failing imported program. I neither replace
        # its assertions nor select a reduced shadow graph.
        original = ROOT / 'tests/token_value_bytes.nano'
        (self.work / 'original-lexer-source.json').write_text(json.dumps(dict(path=str(original),
            sha256=hashlib.sha256(original.read_bytes()).hexdigest()), indent=2) + '\n')
        output = self.work / 'original-lexer.nvm'
        self.command('original-lexer-emit', [ROOT / 'bin/nano_virt', original, '--emit-nvm', '-o', output], timeout=300)
        out, _ = self.command('original-lexer-run', [ROOT / 'bin/nano_vm', output], timeout=30)
        self.assertEqual(out, b'0:4:4\n1:1:2\n2:3:8\n3:3:6\n4:3:2\n5:70:3\n6:7:0\n7:8:0\n8:0:0\n')


if __name__ == '__main__':
    unittest.main()
