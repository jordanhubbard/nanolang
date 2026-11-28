"""I compare ordinary whole-module composition with the actual public C APIs."""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import os
import random
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import run_captured
from tests.test_nanolang_translator_module import corpus as core_corpus, packet
from tests.test_nanolang_translator_payloads import constant, signature, layout, table
from tests.test_nanolang_translator_function_body import corpus as body_corpus
from tests.test_nanolang_translator_stack_heights import ins
from tests.test_nanolang_translator_ownership import record, desc
from tests.test_nanolang_translator_service_modules import corpus as service_corpus

ROOT = Path(__file__).resolve().parents[1]
NONE = 0xffffffff


def assembled(bodies=(b'',), *, params=None, returns=None, locals_=None, upvalues=None,
              declared=None, spans=None, pool=None, strings=(b'function',),
              imports=(), records=0, enums=0, unions=0, extras=(), retain=False, entry=0):
    count = len(bodies)
    params = [[] for _ in bodies] if params is None else params
    returns = [[] for _ in bodies] if returns is None else returns
    locals_ = list(map(len, params)) if locals_ is None else locals_
    upvalues = [0]*count if upvalues is None else upvalues
    declared = [0]*count if declared is None else declared
    if spans is None:
        spans, offset = [], 0
        for code in bodies:
            spans.append((offset, len(code)))
            offset += len(code)
    pool = b''.join(bodies) if pool is None else pool
    shapes = [signature(p, r) for p, r in zip(params, returns)]
    rows = [struct.pack('<IIQQHHHH', 0, i, *span, local, captured, maximum, 0)
            for i, (span, local, captured, maximum) in enumerate(zip(spans, locals_, upvalues, declared))]
    foreign = []
    for p, r, kind, module_name, symbol in imports:
        foreign.append(struct.pack('<IIIB3x', module_name, symbol, len(shapes), kind))
        shapes.append(signature(p, r))
    values = [(2, table(*(constant(5, value) for value in strings))),
              (3, table(*shapes)), (5, table(*rows)), (6, pool)]
    flags = 0
    if records or enums or unions or retain:
        values.append((4, table(*([layout(kind=0)]*records + [layout(kind=3)]*enums + [layout(kind=2)]*unions))))
        flags |= 128
    if imports:
        values.append((8, table(*foreign)))
        flags |= 2
    for kind, data in extras:
        values.append((kind, data))
        flags |= {12:64, 13:256, 14:512}[kind]
    return packet(values, features=flags, entry=entry)


def inherited_body_cases():
    cases, skipped = [], {}
    def skip(reason): skipped[reason] = skipped.get(reason, 0)+1
    for name, data, _ in body_corpus():
        if len(data) < 40:
            skip('incomplete-context'); continue
        start, span, count, current, foreign, linked, strings, records, enums, unions = struct.unpack_from('<10I', data)
        if linked:
            skip('requires-linked-graph'); continue
        if not count or current >= count or count > 64 or foreign > 64:
            skip('nonmodule-context'); continue
        if not strings or strings > 64 or max(records, enums, unions) > 32:
            skip('legacy-count-contexts-not-materialized'); continue
        cursor = 40
        if cursor + count*20 + foreign*8 > len(data):
            skip('incomplete-context'); continue
        functions = [struct.unpack_from('<5I', data, cursor+i*20) for i in range(count)]
        cursor += count*20
        foreign_shapes = [struct.unpack_from('<2I', data, cursor+i*8) for i in range(foreign)]
        cursor += foreign*8
        if any((not results and tag) or max(arity, local, captured) > 65535 or results > 255 or tag > 255
               for arity, results, tag, local, captured in functions):
            skip('not-expressible-signature'); continue
        if any(arity > 65535 or tag > 255 for arity, tag in foreign_shapes):
            skip('not-expressible-import'); continue
        original = data[cursor:]
        pool, spans = bytearray(original), []
        bodies, params, returns, locals_, captures = [], [], [], [], []
        for i, (arity, results, tag, local, captured) in enumerate(functions):
            if i == current:
                spans.append((start, span)); bodies.append(original)
            else:
                # I materialize valid noncurrent bodies so whole-module checks
                # reach the original target rather than an artificial empty
                # result helper. I retain its call/capture signature exactly.
                code = ins('LOAD_GLOBAL', 0)*results + ins('RET')
                spans.append((len(pool), len(code))); pool += code; bodies.append(code)
            params.append([0]*arity)
            returns.append([tag]*results)
            locals_.append(local if i == current else max(local, arity))
            captures.append(captured)
        imports = [([0]*arity, [] if tag == 0 else [tag], 0, 0, 0) for arity, tag in foreign_shapes]
        blob = assembled(bodies, params=params, returns=returns, locals_=locals_, upvalues=captures,
                         spans=spans, pool=bytes(pool), strings=[f'name-{i}'.encode() for i in range(strings)],
                         records=records, enums=enums, unions=unions, imports=imports)
        cases.append(('body-'+name, blob, None, None, False))
    return cases, skipped


def corpus():
    core, private = core_corpus()
    cases = [('core-'+name, data, None, None, False) for name, data, _ in core]
    cases += [('private-'+name, data, 'A', 'A', True) for name, data, _ in private]
    inherited, skipped = inherited_body_cases()
    cases += inherited
    def add(name, data, full=None, declared=None, private=False):
        cases.append((name, data, full, declared, private))
    add('no-functions', packet(), '1:1:', '1:0:')
    add('empty-void', assembled(), '1:1:0', '1:0:0')
    add('empty-int', assembled(returns=[[1]]), '0:I require a body for function[0] with 1 declared result(s).', '1:0:0')
    add('void-result-mismatch-before-empty', assembled(returns=[[0]]), '0:function[0] result signature must be void/0 or non-void/nonzero', '1:0:0')
    add('locals-before-empty', assembled(params=[[1]], returns=[[1]], locals_=[0]), '0:function[0] local_count 0 is smaller than arity 1', '1:0:0')
    add('unknown-zero-declaration', assembled([b'\xff']), 'D', '1:0:0')
    add('unknown-positive-declaration', assembled([b'\xff'], declared=[1]), 'D', 'D')
    add('local-error-positive-declaration', assembled(params=[[1]], locals_=[0], declared=[1]),
        '0:function[0] local_count 0 is smaller than arity 1', '0:function[0] local_count 0 is smaller than arity 1')
    for tag in range(1, 16):
        for count in (1, 2, 255):
            for maximum in (0, 1):
                want = f'0:I require a body for function[0] with {count} declared result(s).'
                add(f'empty-result-{tag}-{count}-{maximum}', assembled(returns=[[tag]*count], declared=[maximum]),
                    want, want if maximum else '1:0:0')
    depth_two = ins('PUSH_I64', 7)*2 + ins('POP') + ins('RET')
    for maximum in (0, 1, 2, 3, 65535):
        error = '0:function 0 declares max_stack 1 but reaches 2'
        add(f'proven-depth-{maximum}', assembled([depth_two], returns=[[1]], declared=[maximum]),
            error if maximum == 1 else '1:1:2',
            '1:0:0' if maximum == 0 else error if maximum == 1 else '1:0:2')
    for first in (b'\xff', ins('POP')):
        label = 'malformed' if first == b'\xff' else 'underflow'
        add(f'global-empty-before-{label}', assembled([first, b''], returns=[[], [1]]),
            '0:I require a body for function[1] with 1 declared result(s).', '1:0:0,0')
        add(f'global-local-before-{label}', assembled([first, b''], params=[[], [1]], locals_=[0, 0]),
            '0:function[1] local_count 0 is smaller than arity 1', '1:0:0,0')
        add(f'positive-later-before-{label}', assembled([first, depth_two], returns=[[], [1]], declared=[0, 1]),
            '0:function 1 declares max_stack 1 but reaches 2', '0:function 1 declares max_stack 1 but reaches 2')
    add('positive-later-skips-unknown', assembled([b'\xff', depth_two], returns=[[], [1]], declared=[0, 2]),
        'D', '1:0:0,2')
    for maximum in (0, 1):
        add(f'overlap-before-body-{maximum}', assembled([b'', b''], pool=b'\0\0', spans=[(0, 2), (1, 1)], declared=[0, maximum]),
            '0:function[1] code range overlaps function[0]',
            '0:function[1] code range overlaps function[0]' if maximum else '1:0:0,0')
        add(f'empty-span-shared-start-{maximum}', assembled([b'', b'', b''], pool=b'\0\0', spans=[(0, 1), (1, 0), (1, 1)], declared=[maximum]*3),
            '1:1:0,0,0', '1:0:0,0,0')
        add(f'reverse-adjacent-spans-{maximum}', assembled([b'', b''], pool=b'\0\0', spans=[(1, 1), (0, 1)], declared=[maximum]*2),
            '1:1:0,0', '1:0:0,0')
        for arity in (16, 17, 65535):
            blob = assembled([b'\xff'], declared=[maximum], imports=[([1]*arity, [], 0, 0, 0)])
            error = f'0:import[0] param_count {arity} exceeds the foreign-call argument limit of 16'
            add(f'import-before-body-{arity}-{maximum}', blob, error if arity > 16 else 'D',
                (error if arity > 16 else 'D') if maximum else '1:0:0')
    owner = record(1, functions=[(0, 0, desc(), [])])
    for maximum in (0, 1):
        add(f'ownership-remains-deferred-{maximum}', assembled(declared=[maximum], retain=True, extras=[(13, owner)]),
            None, 'A' if maximum else '1:0:0', private=True)
    for name, data, _ in service_corpus():
        if name.endswith('-valid'):
            add('service-policy-'+name, data, '0:I refuse service contracts before mixed execution selection')
    # I retain independent multiple-function bodies and error priority, rather
    # than merely wrapping every existing one-function fixture.
    rng = random.Random(114275)
    for i in range(256):
        bodies = []
        for _ in range(rng.randrange(1, 5)):
            code = bytearray()
            for _ in range(rng.randrange(0, 12)):
                code += rng.choice([ins('PUSH_I64', rng.randrange(-20, 21)), ins('POP'), ins('NOP'), ins('DUP'), ins('I64_ADD')])
            code += ins('RET')
            bodies.append(bytes(code))
        add(f'multiple-functions-{i}', assembled(bodies, declared=[rng.choice([0, 0, 1, 2, 8]) for _ in bodies]))
    return cases, skipped


class NanoLangTranslatorModuleVerifier(unittest.TestCase):
    def test_public_declared_and_full_pipeline(self):
        retained = os.environ.get('NANOLANG_MODULE_VERIFIER_PRODUCTS')
        with (nullcontext(retained) if retained else tempfile.TemporaryDirectory(prefix='nano-module-verifier-')) as name:
            directory = Path(name)
            directory.mkdir(parents=True, exist_ok=True)
            inputs = directory/'inputs'
            inputs.mkdir(exist_ok=False)
            cases, skipped = corpus()
            paths = []
            for i, (_, blob, _, _, _) in enumerate(cases):
                path = inputs/f'{i}.nvm'; path.write_bytes(blob); paths.append(path)
            oracle = ROOT/'obj/nvm2c_module_verifier_reference'
            routes = build_routes(ROOT/'tests/nvm2c_module_verifier_driver.nano', directory)
            command_index = []
            command_directory = directory/'commands'
            command_directory.mkdir()
            def reports(command, selected, declared=False):
                result = []
                for start in range(0, len(selected), 32):
                    args = [*command, *(['--declared'] if declared else []), *selected[start:start+32]]
                    completed = run_captured(args, timeout=600, cwd=ROOT)
                    index = len(command_index)
                    (command_directory/f'{index:05d}.stdout').write_bytes(completed.stdout)
                    (command_directory/f'{index:05d}.stderr').write_bytes(completed.stderr)
                    command_index.append(dict(argv=list(map(str,args)), returncode=completed.returncode))
                    self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors='replace'))
                    result += completed.stdout.decode().splitlines()
                self.assertEqual(len(result), len(selected))
                return result
            counts = {}
            for declared, column, mode in [(False, 2, 'full'), (True, 3, 'declared')]:
                reference = reports([oracle], paths, declared)
                (directory/f'{mode}-reference.txt').write_text('\n'.join(reference)+'\n')
                for case, actual in zip(cases, reference):
                    if case[column] is not None and not case[4]:
                        with self.subTest(reference=case[0], mode=mode): self.assertEqual(actual, case[column])
                baseline = None
                for route, command in routes:
                    actual = reports(command, paths, declared)
                    (directory/f'{mode}-{route}.txt').write_text('\n'.join(actual)+'\n')
                    for case, wanted, value in zip(cases, reference, actual):
                        with self.subTest(route=route, mode=mode, case=case[0]):
                            if value == 'A' and case[4]: pass
                            else: self.assertEqual(value, wanted)
                            if case[column] is not None: self.assertEqual(value, case[column])
                    if baseline is None: baseline = actual
                    else: self.assertEqual(actual, baseline, route)
                counts[mode] = {prefix:sum(row.startswith(prefix) for row in baseline) for prefix in ['1:', '0:', 'R:', 'D', 'A']}
            producers = [([ROOT/'bin/nano_virt'], [])]
            source = source_compiler_command()
            if source: producers.append((source, ['--emit-nvm']))
            products = []
            for i, (producer, flags) in enumerate(producers):
                for original in ['examples/language/nl_functions_basic.nano', 'examples/language/nl_control_if_while.nano', 'examples/cross_backend/hello_cross_backend.nano']:
                    path = directory/f'original-{i}-{Path(original).stem}.nvm'
                    checked([*producer, ROOT/original, *flags, '-o', path], build=True)
                    products.append(path)
            for mode in [False, True]:
                expected = reports([oracle], products, mode)
                self.assertTrue(all(value.startswith('1:') for value in expected))
                (directory/f'original-{mode}-reference.txt').write_text('\n'.join(expected)+'\n')
                for route, command in routes:
                    actual = reports(command, products, mode)
                    self.assertEqual(actual, expected, (route, mode))
                    (directory/f'original-{mode}-{route}.txt').write_text('\n'.join(actual)+'\n')
            manifest = dict(cases=len(cases), modes=counts, routes=[name for name, _ in routes],
                            skipped_legacy_contexts=skipped, original_products=len(products),
                            independent_expectations={mode:sum(case[column] is not None and not case[4] for case in cases) for mode,column in [('full',2),('declared',3)]},
                            inputs=[dict(name=case[0], file=path.name, sha256=hashlib.sha256(case[1]).hexdigest()) for case,path in zip(cases, paths)])
            (directory/'commands.json').write_text(json.dumps(command_index, indent=2)+'\n')
            (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(f'I matched {len(cases)} actual declared/full module reports through {len(routes)} routes and {len(products)} original source products; excluded legacy-only contexts remain counted separately.')
