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
        common = [*cls.cc, *cls.flags, '-std=c99', '-Wall', '-Wextra', '-Werror', '-I', ROOT / 'src']
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
        out, _ = self.command('checked-lifetimes', [self.owned])
        self.assertIn(b'checked ownership assertions', out)
        self.command('checked-callable-hook', [self.programs, 'callable'])

    def test_parsed_evaluator_and_public_escape(self):
        for name in ('mutations', 'staging', 'imported', 'escape', 'async'):
            self.command('evaluator-' + name, [self.programs, 'escape' if name == 'escape' else 'program', FIXTURES / (name + '.nano')])

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

    def source_routes(self, name, source, reject=False, vm=True):
        path = self.work / (name + '.nano'); path.write_text(source)
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            output = self.work / (name + '-' + compiler); output.write_bytes(SENTINEL)
            out, err = self.command(name + '-' + compiler + '-build',
                [ROOT / 'bin' / compiler, path, '-o', output, '--keep-c'], expected=tuple(range(1, 126)) if reject else (0,), timeout=300)
            if reject:
                self.assertEqual(output.read_bytes(), SENTINEL)
                self.assertNotIn(b'C compilation failed', out + err)
            else: self.command(name + '-' + compiler + '-run', [output], timeout=15)
        # I exercise checker refusal in the actual evaluator too, before run_program.
        self.command(name + '-evaluator', [self.programs, 'reject' if reject else 'program', path])
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
