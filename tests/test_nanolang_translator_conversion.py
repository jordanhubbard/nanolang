"""I compare reusable conversion facts with the actual C reverse converter."""
from contextlib import nullcontext
from pathlib import Path
import json
import os
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.test_nanolang_translator_module import corpus as core_corpus, packet
from tests.test_nanolang_translator_payloads import constant, signature, layout, table
from tests.test_nanolang_translator_ownership_module import corpus as ownership_corpus
from tests.test_nanolang_translator_service_modules import corpus as service_corpus
from tests.test_nanolang_translator_passive_module import corpus as passive_corpus

ROOT = Path(__file__).resolve().parents[1]
NONE = 0xffffffff


def private_ownership_routes(directory):
    """I consult the actual C route classifier before allowing private deferral."""
    directory.mkdir()
    carriers = []
    selected = set()
    for name, _, _, private, carrier in ownership_corpus():
        if private:
            selected.add(name)
        if carrier is not None:
            path = directory/f'{len(carriers)}.bin'
            path.write_bytes(carrier)
            carriers.append((name, path))
    records = []
    for start in range(0, len(carriers), 32):
        batch = carriers[start:start+32]
        rows = checked([ROOT/'obj/nvm2c_declaration_reference', '0',
                        *[path for _, path in batch]], timeout=600).splitlines()
        if len(rows) != len(batch):
            raise AssertionError('I require one actual C routing report per carrier.')
        for (name, _), row in zip(batch, rows):
            if not row.startswith('R|'):
                raise AssertionError(row)
            if int(row[2:].split(',')[2]) == 1:
                selected.add(name)
            records.append({'case': name, 'actual_c_route': row})
    (directory/'reports.json').write_text(json.dumps(records, indent=2)+'\n')
    return selected


def corpus(widths, private_ownership=()):
    core, pending = core_corpus()
    cases = [(name, data, None, False) for name, data, _ in core]
    cases += [(name, data, -1, True) for name, data, _ in pending]
    cases += [('ownership-' + name, data, -1 if deferred else None, deferred or name in private_ownership)
              for name, data, _, deferred, _ in ownership_corpus()]
    cases += [('service-' + name, data, None, False)
              for name, data, _ in service_corpus()]
    cases += [('passive-' + name, data, None, False)
              for name, data, _ in passive_corpus(widths)]

    def add(name, *, names=(b'main',), params=(), returns=(), debug=None,
            layouts=None, code=b'', span=None, locals_=None, upvalues=0, expected=None):
        values = [(2, table(*(constant(5, name) for name in names))),
                  (3, table(signature(params, returns))),
                  (5, table(struct.pack('<IIQQHHHH', 0, 0, 0,
                                        len(code) if span is None else span,
                                        len(params) if locals_ is None else locals_, upvalues, 0, 0))),
                  (6, code)]
        flags = 0
        if debug is not None:
            values.append((10, table(struct.pack('<QII', debug, 7, 11))))
            flags += 8
        if layouts is not None:
            values.append((4, layouts))
            flags += 128
        cases.append((name, packet(values, features=flags, entry=0), expected, False))

    # These exact facts are independent expectations, beyond C agreement.
    add('empty-void-is-still-only-conversion', expected='0:::0:0:0:1:6d61696e:1:0,0,0,0,0,0,0')
    add('empty-result-is-not-executable-admission', returns=[1],
        expected='0:::0:0:0:1:6d61696e:1:0,0,0,1,0,0,1')
    add('nonopcode-is-not-body-admission', code=b'\xff',
        expected='0:ff::0:0:0:1:6d61696e:1:0,1,0,0,0,0,0')
    add('signature-order-and-padding', names=(b'main', b'\0\xff', b''),
        params=[15, 0, 7], returns=[14, 1], locals_=9, upvalues=5,
        code=b'\0\xff', layouts=table(layout(kind=0), layout(kind=1), layout(kind=2), layout(kind=3))+b'tail',
        expected='0:00ff:0400000000000000ffffffff01000000ffffffff02000000ffffffff03000000ffffffff:1:1:1:3:6d61696e:00ff::1:0,2,3,2,9,5,14,15,0,7')
    add('distinct-byte-lengths', names=(b'main', b'a', b'a\0', b'a\0b'),
        expected='0:::0:0:0:4:6d61696e:61:6100:610062:1:0,0,0,0,0,0,0')
    add('duplicate-before-debug', names=(b'main', b'main'), debug=2**32, expected='13')
    add('debug-u32-maximum', debug=2**32-1,
        expected='0:::0:0:0:1:6d61696e:1:0,0,0,0,0,0,0')
    add('debug-too-wide', debug=2**32, expected='13')
    add('debug-negative-signed-carrier', debug=2**63, expected='13')
    for arity in (0, 1, 2, 3, 4, 17):
        for count in (0, 1, 2, 255, 256):
            for tag in range(16):
                params = [(tag+i) % 16 for i in range(arity)]
                add(f'signature-{arity}-{count}-{tag}', params=params,
                    returns=[tag]*count, upvalues=tag,
                    expected='13' if count > 255 else None)
    return cases


class NanoLangTranslatorConversion(unittest.TestCase):
    def test_complete_projected_facts_and_original_products(self):
        retained = os.environ.get('NANOLANG_TRANSLATOR_CONVERSION_PRODUCTS')
        context = nullcontext(retained) if retained else tempfile.TemporaryDirectory(prefix='nano-conversion-')
        with context as name:
            directory = Path(name)
            directory.mkdir(parents=True, exist_ok=True)
            self.assertFalse((directory / 'inputs').exists(), 'I require a fresh product directory.')
            inputs = directory / 'inputs'
            inputs.mkdir()
            oracle = ROOT/'obj/nvm2c_module_reference'
            widths = list(map(int, checked([oracle, '--widths']).splitlines()))
            selected = private_ownership_routes(directory/'routing')
            cases = corpus(widths, selected)
            paths = []
            for i, (_, data, _, _) in enumerate(cases):
                path = inputs / f'{i}.nvm'
                path.write_bytes(data)
                paths.append(path)
            routes = build_routes(ROOT/'tests/nvm2c_module_driver.nano', directory)
            oracle = ROOT/'obj/nvm2c_module_reference'

            def reports(command, selected):
                rows = []
                for start in range(0, len(selected), 64):
                    rows += checked([*command, '--conversion', *selected[start:start+64]], timeout=120).splitlines()
                self.assertEqual(len(rows), len(selected))
                return rows

            reference = reports([oracle], paths)
            (directory/'reference.txt').write_text('\n'.join(reference)+'\n')
            for (case, _, expected, pending), actual in zip(cases, reference):
                if expected is not None and not pending:
                    with self.subTest(reference=case): self.assertEqual(actual, expected)
            baseline = None
            deferred_counts = []
            for route, command in routes:
                actual = reports(command, paths)
                (directory/f'{route}.txt').write_text('\n'.join(actual)+'\n')
                deferred_counts.append(sum(value == '-1' for value in actual))
                for (case, _, declared, pending), wanted, value in zip(cases, reference, actual):
                    with self.subTest(route=route, case=case):
                        if value == '-1':
                            self.assertTrue(pending, 'I require an independently classified private profile.')
                        else:
                            self.assertNotEqual(declared, -1, 'I require this declared private boundary to remain deferred.')
                            self.assertEqual(value, wanted)
                if baseline is None:
                    baseline = actual
                else:
                    self.assertEqual(actual, baseline, route)
            self.assertEqual(len(set(deferred_counts)), 1)
            producers = [([ROOT/'bin/nano_virt'], [])]
            source = source_compiler_command()
            if source:
                producers.append((source, ['--emit-nvm']))
            products = []
            for i, (producer, flags) in enumerate(producers):
                for original in ('examples/language/nl_functions_basic.nano',
                                 'examples/language/nl_control_if_while.nano',
                                 'examples/cross_backend/hello_cross_backend.nano'):
                    product = directory/f'original-{i}-{Path(original).stem}.nvm'
                    checked([*producer, ROOT/original, *flags, '-o', product], build=True)
                    products.append(product)
            original_reference = reports([oracle], products)
            self.assertTrue(all(value.startswith('0:') for value in original_reference))
            (directory/'original-reference.txt').write_text('\n'.join(original_reference)+'\n')
            for route, command in routes:
                actual = reports(command, products)
                (directory/f'original-{route}.txt').write_text('\n'.join(actual)+'\n')
                self.assertEqual(actual, original_reference, route)
            manifest = {'cases': len(cases), 'private_candidates': sum(case[3] for case in cases),
                        'pending': deferred_counts[0],
                        'independent_expectations': sum(case[2] is not None and not case[3] for case in cases),
                        'original_products': len(products), 'routes': [route for route, _ in routes],
                        'labels': [case[0] for case in cases]}
            (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(f'I matched {len(cases)} conversion reports and {len(products)} original products through {len(routes)} routes; {manifest["pending"]} private profiles remain deferred.')
