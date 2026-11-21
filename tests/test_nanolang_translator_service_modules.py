"""I preserve shared service framing, rich preflight and descriptive claims in complete modules."""
from pathlib import Path
import copy
import json
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.test_nanolang_translator_module import packet
from tests.test_nanolang_translator_container import patched
from tests.test_nanolang_translator_payloads import constant, signature, table
from tests.test_nanolang_translator_service_v2 import view
from tests.test_nanolang_translator_service_nominal import producer_cases
from tests.test_nanolang_translator_ownership import record

ROOT = Path(__file__).resolve().parents[1]
NONE = 0xffffffff


def sections(model):
    constants = table(*(constant(tag, data) for tag, data in model['constants']))
    signatures = table(*(signature(params, returns) for _, _, params, returns in model['signatures']))
    functions = table(*(struct.pack('<IIQQHHHH', 0, sig, start, length, locals_, 0, 0, 0)
                        for sig, locals_, start, length in model['functions']))
    imports = table(*(struct.pack('<IIIB3x', *row) for row in model['imports']))
    values = {2: constants, 3: signatures, 4: model['layouts'], 5: functions,
              6: model['code'], 8: imports}
    if model['ownership']: values[13] = model['ownership']
    if model['service']: values[14] = model['service']
    if model['links']: values[9] = table(*(bytes(16) for _ in range(model['links'])))
    if model['callbacks']: values[11] = table(*(bytes(16) for _ in range(model['callbacks'])))
    return values


def complete(model, *, changes=None, order=None, features=None, entry=NONE):
    values = sections(model) | (changes or {})
    if order is not None: values = {kind: values[kind] for kind in order}
    if features is None:
        features = 2 + 128 + (256 if model['ownership'] else 0) + (512 if model['service'] else 0)
    return packet(list(values.items()), features=features, entry=entry)


def corpus():
    cases = []
    for label, cats, legacy in [('legacy', (1,), True), ('file', (1,), False),
                                ('tcp', (2,), False), ('multi', (1, 2, 1, 3), False)]:
        base = view(cats, legacy=legacy)
        if legacy: base['ownership'] = b''
        for function in base['functions']: function[2:] = [0, len(base['code'])]
        def add(name, model=None, expected=None, **options):
            cases.append((label + '-' + name, complete(base if model is None else model, **options), expected))
        add('valid', expected=0)
        values = sections(base)
        add('reverse-directory', expected=0, order=list(reversed(values)))
        flags = 2 + 128 + (256 if base['ownership'] else 0) + 512
        for bit in (2, 128, 256, 512):
            if flags & bit:
                add(f'missing-feature-{bit}', features=flags ^ bit, expected=14)
        add('service-feature-before-bad-function-name', changes={5: patched(values[5], 4, '<I', NONE)}, features=flags ^ 512, expected=14)
        add('bad-function-name', changes={5: patched(values[5], 4, '<I', NONE)}, expected=13)
        for size in (0, 1, 55, 57, 79, 104, 119, 121, 127, 129, 4113):
            add(f'bad-service-frame-{size}', changes={14: bytes(size)}, expected=9)
        for first in (False, True):
            changed = values | {2: b'', 14: b'\0'}
            order = [14, 2] if first else [2, 14]
            order += [kind for kind in changed if kind not in order]
            add('directory-error-priority-' + str(first), changes=changed, order=order, expected=9 if first else 1)
        for offset in range(len(base['service'])):
            raw = bytearray(base['service']); raw[offset] ^= 0x80
            add(f'service-bit-{offset}', changes={14: bytes(raw)})
        for field in (0, 1, 2):
            model = copy.deepcopy(base); model['imports'][0][field] = NONE
            add(f'bad-import-{field}', model=model, expected=13 if legacy and field == 2 else 8)
        model = copy.deepcopy(base); model['constants'].append([1, bytes(8)])
        add('unused-nonstring-narrowing', model=model, expected=8)
        add('bad-passive', changes={12: b'bad'}, features=flags | 64, expected=13)
        add('bad-passive-feature-first', changes={12: b'bad'}, expected=14)
        add('owner-version-and-passive', changes={13: record(4), 12: b'bad'}, features=flags | 256 | 64, expected=3 if legacy else 8)
        for start, length in ((0, 0), (2**32, 0), (2**63, 0), (0, 2**64 - 1)):
            model = copy.deepcopy(base)
            model['functions'][0][2:] = [start, length]
            add(f'function-span-{start}-{length}', model=model, expected=0 if start == length == 0 else 13)
    cases.append(('public-websocket-has-no-private-codec', complete(view((3,))), 9))
    return cases


class SharedServiceModules(unittest.TestCase):
    def test_whole_module_service_order_and_claim_bytes(self):
        cases = corpus()
        with tempfile.TemporaryDirectory(prefix='shared-service-modules-') as name:
            work = Path(name); paths = []
            products = work / 'products'
            producer_cases(products)
            records = json.loads((products / 'records.json').read_text())
            self.assertEqual(len(records), 8 if source_compiler_command() else 4)
            for item in records:
                expected = 9 if item['profile'] == 4 else 0
                cases.append(('source-' + item['module'],
                              (products / item['module']).read_bytes(), expected))
            for index, (_, data, _) in enumerate(cases):
                path = work / f'{index}.nvm'; path.write_bytes(data); paths.append(path)
            def execute(command):
                reports = []
                for start in range(0, len(paths), 32):
                    reports.extend(checked([*command, '--services', *paths[start:start + 32]], timeout=600).splitlines())
                self.assertEqual(len(reports), len(cases)); return reports
            expected = execute([ROOT / 'obj/nvm2c_module_reference'])
            for (label, _, status), actual in zip(cases, expected):
                if status is not None:
                    with self.subTest(oracle=label): self.assertEqual(int(actual.split(':')[0]), status)
            routes = build_routes(ROOT / 'tests/nvm2c_module_driver.nano', work)
            for label, command in routes:
                actual = execute(command)
                for (case, _, _), want, got in zip(cases, expected, actual):
                    with self.subTest(route=label, case=case): self.assertEqual(got, want)
            print(f'I matched {len(cases)} complete shared service decisions and exact service/ownership/passive bytes across {len(routes)} routes.')


if __name__ == '__main__':
    unittest.main()
