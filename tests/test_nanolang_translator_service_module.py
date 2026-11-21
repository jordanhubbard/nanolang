"""I compare shared service metadata and classification without execution grants."""
from pathlib import Path
import copy
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked
from tests.test_nanolang_translator_service_nominal import corpus as nominal_corpus, model, packet, producer_cases
from tests.test_nanolang_translator_ownership import function, NONE
from tests.test_nanolang_translator_services import fixed

ROOT = Path(__file__).resolve().parents[1]


def corpus():
    cases = []
    for profile, name, data, status in nominal_corpus():
        expected = None if status is None else (8 if profile == 4 else {0: 0, 1: 8, 2: 13}[status])
        cases.append((f'nominal-{profile}-{name}', data, expected))

    def add(name, value, status):
        cases.append((name, packet(value, 1), status))

    base = model([1])
    base['service'] = fixed(1, list(range(5)), [])
    add('legacy-valid', base, 0)
    reversed_model = model([1], reverse=True)
    reversed_model['service'] = fixed(1, reversed_model['bindings'][0][1][:5], [])
    add('legacy-reversed-imports', reversed_model, 0)
    for n in range(len(base['service'])):
        m = copy.deepcopy(base); m['service'] = base['service'][:n]
        add(f'legacy-truncation-{n}', m, 8)
    m = copy.deepcopy(base); m['service'] += b'\0'; add('legacy-tail', m, 8)
    for offset in (0, 2, 4, 6, 8):
        for value in range(256):
            m = copy.deepcopy(base); raw = bytearray(m['service']); raw[offset] = value; m['service'] = bytes(raw)
            add(f'legacy-header-{offset}-{value}', m, 0 if value == base['service'][offset] else 8)
    for count in (0, 1, 4, 6):
        m = copy.deepcopy(base); m['imports'] = (m['imports'] + m['imports'])[:count]
        add(f'legacy-import-count-{count}', m, 14)
    for key in ('refs', 'callbacks'):
        m = copy.deepcopy(base); m[key] = 1; add(f'legacy-{key}', m, 14)
        m['service'] = b''; add(f'legacy-raw-before-{key}', m, 8)
    for method in range(5):
        for field, value in [(0, NONE), (1, NONE), (2, 0), (3, 0), (4, 1)]:
            m = copy.deepcopy(base); m['imports'][method][field] = value
            if field == 3: m['imports'][method][5] = b''
            good = field == 3 and method == 0
            add(f'legacy-import-{method}-{field}', m, 0 if good else 8)
        for index in (5, 65535, NONE):
            m = copy.deepcopy(base); indices = list(range(5)); indices[method] = index
            m['service'] = fixed(1, indices, [])
            add(f'legacy-index-{method}-{index}', m, 8 if index == NONE else 13)
    for tag in range(256):
        m = copy.deepcopy(base); m['imports'][0][4] = tag
        add(f'legacy-result-{tag}', m, 0 if tag == 10 else 8)
        m = copy.deepcopy(base); m['imports'][1][5] = bytes([tag, 1])
        add(f'legacy-file-param-{tag}', m, 0 if tag == 8 else 8)
        m = copy.deepcopy(base); m['imports'][1][5] = bytes([8, tag])
        add(f'legacy-int-param-{tag}', m, 0 if tag == 1 else 8)
    for key in ('layouts', 'ownership'):
        for raw in (b'', b'\xff', struct.pack('<I', 65537)):
            m = copy.deepcopy(base); m[key] = raw
            add(f'legacy-ignores-{key}-{len(raw)}', m, 0)
    for n in (0, 1, 2, 3, 4, 5):
        m = copy.deepcopy(base); m['service'] = bytes([2, 0, 3, 0, 0][:n])
        add(f'nominal-prefix-no-legacy-fallback-{n}', m, 8)
    m = copy.deepcopy(base); m['service'] = b''; m['imports'] = []; m['functions'] = []
    add('absent', m, 0)
    m['refs'] = 1; m['callbacks'] = 1; add('absent-unrelated-claims', m, 0)
    for kind in (0, 1, 2, 3, 255):
        m = copy.deepcopy(base); m['service'] = b''
        for entry in m['imports']: entry[2] = kind
        add(f'presence-import-kind-{kind}', m, 8 if kind == 3 else 0)
    # I scan only checked function extents, stop at unknown/truncated opcodes,
    # and recognize File claims before reading their operands.
    for op in range(256):
        for context in ('instruction', 'integer-operand', 'after-unknown'):
            m = copy.deepcopy(base); m['service'] = b''; m['imports'] = []
            m['code'] = bytes([op]) if context == 'instruction' else (bytes([1, op])+bytes(7) if context == 'integer-operand' else bytes([255, op]))
            m['functions'] = [function(start=0, length=len(m['code']))]
            add(f'scan-{context}-{op}', m, 8 if context == 'instruction' and 145 <= op <= 152 else 0)
    for start, length, status in [(0,1,8),(0,0,0),(1,1,0),(1,0,0),(NONE,1,0),(0,NONE,0)]:
        m = copy.deepcopy(base); m['service'] = b''; m['imports'] = []; m['code'] = bytes([145]); m['functions'] = [function(start=start,length=length)]
        add(f'scan-span-{start}-{length}', m, status)
    return cases


class ServiceModule(unittest.TestCase):
    def compare(self, commands):
        cases = corpus()
        for profile, name, data, _ in producer_cases():
            cases.append((name, data, 8 if profile == 4 else 0))
        with tempfile.TemporaryDirectory(prefix='nano-service-module-') as directory:
            work = Path(directory)
            paths = []
            for i, (_, data, _) in enumerate(cases):
                path = work / f'{i}.bin'; path.write_bytes(data); paths.append(path)
            routes = commands(work)
            def execute(command):
                output = []
                for offset in range(0, len(paths), 32):
                    output += checked([*command, '0', *paths[offset:offset+32]], timeout=600).splitlines()
                self.assertEqual(len(output), len(cases))
                return output
            reference = execute([ROOT / 'obj/nvm2c_service_nominal_reference'])
            for (name, _, status), report in zip(cases, reference):
                if status is not None:
                    with self.subTest(oracle=name): self.assertEqual(int(report.split('|')[0]), status)
                if name.startswith('scan-'):
                    with self.subTest(classification=name): self.assertEqual(report, f'{status}|0,{int(status == 8)},{int(status == 8)}')
                if name in ('absent', 'absent-unrelated-claims') or name.startswith('presence-import-kind-'):
                    claim = int(name == 'presence-import-kind-3')
                    with self.subTest(classification=name): self.assertEqual(report, f'{status}|{claim},0,{claim}')
            for name, command in routes:
                actual = execute(command)
                for (case, _, _), wanted, got in zip(cases, reference, actual):
                    with self.subTest(route=name, case=case): self.assertEqual(got, wanted)
        return len(cases), len(routes)

    def test_complete_shared_statuses_and_classification(self):
        count, routes = self.compare(lambda work: build_routes(ROOT / 'tests/nvm2c_service_nominal_driver.nano', work))
        print(f'I matched {count} complete shared service status/presence/File/pending reports through {routes} routes; no execution authority follows.')

    def test_actual_shared_providers_sanitizers(self):
        count, _ = self.compare(lambda work: [('actual-shared-asan-ubsan', [ROOT / 'obj/nvm2c_service_module_reference_asan'])])
        print(f'I matched {count} complete reports through instrumented shared providers.')


if __name__ == '__main__': unittest.main()
