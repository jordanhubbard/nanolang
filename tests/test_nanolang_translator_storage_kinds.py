"""I compare every physical storage-helper input without granting admission."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked, source_compiler_command
from tests.native_process import native_link_flags

ROOT = Path(__file__).resolve().parents[1]


def packet(rows):
    result = bytearray(struct.pack('<I', len(rows)))
    for op, left, right in rows:
        result.append(op)
        if op == 2:
            result.extend(struct.pack('<H', left))
        else:
            result.append(left)
            if op == 1:
                result.append(right)
    return bytes(result)


def complete_packets():
    groups = [('kinds', [(0, k, 0) for k in range(256)]),
              ('fields', [(1, a, b) for a in range(256) for b in range(256)]),
              ('tags', [(2, k, 0) for k in range(65536)]),
              ('opcodes', [(3, k, 0) for k in range(256)])]
    return [(f'{name}-{i // 4096:02d}', packet(rows[i:i + 4096]))
            for name, rows in groups for i in range(0, len(rows), 4096)]


class StorageKinds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='nano-storage-kinds-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory = Path(cls.workspace.name)
        source = (ROOT / 'src/nanoisa/nvm2c.c').read_bytes()
        begin = b'#define NVM2C_VK_INT 0'
        end = b'typedef struct Nvm2cFieldBlock'
        assert source.count(begin) == source.count(end) == 1
        actual = source[source.index(begin):source.index(end)]
        assert actual.count(b'static ') == 11
        (cls.directory / 'nvm2c_storage_actual.h').write_bytes(actual)
        (cls.directory / 'reference-source.json').write_text(json.dumps({
            'source': 'src/nanoisa/nvm2c.c', 'source_sha256': hashlib.sha256(source).hexdigest(),
            'verbatim_helper_sha256': hashlib.sha256(actual).hexdigest(),
            'begin_marker': begin.decode(), 'end_marker': end.decode(), 'functions': 11,
            'product_source_edited': False}, indent=2) + '\n')
        cls.references = []
        for name, flags in [('ordinary', ['-O2']),
                            ('sanitized', ['-O1', '-g', '-fsanitize=address,undefined',
                                           '-fno-sanitize-recover=all'])]:
            binary = cls.directory / ('reference-' + name)
            checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-Wall', '-Wextra',
                     '-Werror', *flags, '-I', cls.directory,
                     ROOT / 'tests/nvm2c_storage_kinds_reference.c',
                     *native_link_flags(), '-o', binary], build=True)
            cls.references.append((name, [binary]))
        cls.fixture = ROOT / 'tests/nvm2c_storage_kinds_driver.nano'
        cls.routes = build_routes(cls.fixture, cls.directory)
        assert len(cls.routes) == (5 if source_compiler_command() else 3)

    def compare(self, rows, label):
        paths = []
        manifest = {'qualification': False, 'cases': [], 'routes': []}
        for name, data in rows:
            path = self.directory / (label + '-' + name + '.bin')
            path.write_bytes(data)
            paths.append(path)
            manifest['cases'].append({'name': name, 'path': str(path),
                                      'sha256': hashlib.sha256(data).hexdigest()})
        baseline = None
        for name, command in [*self.references, *self.routes]:
            result = checked([*command, *paths], timeout=60)
            if baseline is None:
                baseline = result
            else:
                self.assertEqual(result, baseline, name)
            raw = result.encode()
            (self.directory / (label + '-' + name + '.reports')).write_bytes(raw)
            manifest['routes'].append({'name': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                                       'lines': len(result.splitlines())})
        (self.directory / (label + '-manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
        return baseline

    def test_complete_physical_domains(self):
        result = self.compare(complete_packets(), 'complete')
        parsed = [line.split('|') for line in result.splitlines()]
        kinds = {int(r[1]): tuple(map(int, r[2:])) for r in parsed if r[0] == 'K'}
        fields = {(int(r[1]), int(r[2])): tuple(map(int, r[3:])) for r in parsed if r[0] == 'F'}
        tags = {int(r[1]): tuple(map(int, r[2:])) for r in parsed if r[0] == 'T'}
        opcodes = {int(r[1]): int(r[2]) for r in parsed if r[0] == 'B'}
        self.assertEqual((len(kinds), len(fields), len(tags), len(opcodes)), (256, 65536, 65536, 256))
        self.assertEqual(sum(int(r[1]) for r in parsed if r[0] == 'Z'), 131584)
        self.assertEqual({k for k, r in kinds.items() if r[0]}, set(range(65, 128)))
        self.assertEqual({k for k, r in kinds.items() if r[2]}, {0, 1, 9, 11})
        self.assertEqual({k for k, r in kinds.items() if r[3]}, {3, 10, 12, 14, 15, 16, 17})
        self.assertEqual({k for k, r in kinds.items() if r[4]}, {3, 5, 10, 12, 15})
        for k, bits in [(0, 65), (9, 66), (11, 68), (3, 96), (2, 72), (255, 72)]:
            self.assertEqual(kinds[k][1], bits)
        for (left, right), (unboxes, converts, accepted, merged) in fields.items():
            self.assertEqual(accepted, fields[right, left][2])
            if accepted:
                self.assertEqual(merged, fields[right, left][3])
            else:
                self.assertEqual(merged, left)
            if unboxes:
                self.assertEqual(right, 8)
                self.assertTrue(converts)
                self.assertIn(left, {3, 5, 10, 12, 14, 15, 16, 17})
        self.assertEqual(fields[65, 96], (0, 1, 1, 97))
        self.assertEqual(fields[0, 11], (0, 0, 0, 0))
        self.assertEqual(fields[6, 8], (0, 0, 0, 6))
        self.assertEqual(sum(r[0] for r in tags.values()), 126)
        self.assertEqual(sum(r[1] for r in tags.values()), 254)
        self.assertTrue(all(not scalar or payload for scalar, payload in tags.values()))
        for mask, expected in [(16384, (0, 0)), (16385, (0, 0)), (16386, (1, 1)),
                               (16512, (0, 1)), (16448, (0, 0)), (65535, (0, 0))]:
            self.assertEqual(tags[mask], expected)
        self.assertEqual({k for k, r in opcodes.items() if r},
                         {3, 40, 41, 42, 43, 44, 45, 48, 49, 50, 67, 68, 75, 76, 123, 138, 140,
                          208, 209, 210, 211, 212, 213, 216, 217, 218, 219, 220, 221, 224, 225, 226})

    def test_malformed_and_prior_packet_boundaries(self):
        sample = packet([(0, 0, 0), (1, 65, 96), (2, 16386, 0), (3, 3, 0)])
        cases = [(f'truncated-{n}', sample[:n]) for n in range(len(sample))]
        cases += [('unknown-op', struct.pack('<I', 1) + bytes([4, 0])),
                  ('late-unknown', struct.pack('<I', 2) + bytes([0, 0, 4, 0])),
                  ('huge-count', struct.pack('<I', 0xffffffff)),
                  ('count-cap', struct.pack('<I', 262145)), ('trailing', sample + b'\0'),
                  ('size-cap', b'\0' * 1048577)]
        self.assertEqual(self.compare(cases, 'malformed'), 'P\n' * len(cases))
        neighbors = [('before', packet([(0, 0, 0)])), ('late', struct.pack('<I', 2) + bytes([0, 0, 4, 0])),
                     ('after', packet([(3, 3, 0)])), ('empty', packet([]))]
        self.assertEqual(self.compare(neighbors, 'neighbors'), 'Z|1\nK|0|0|65|1|0|0\nP\nZ|1\nB|3|1\nZ|0\n')

    def test_imported_shadows(self):
        binary = self.directory / 'imported-native'
        checked([ROOT / 'bin/nanoc_c', self.fixture, '--test-imports', '-o', binary], build=True)
        self.assertEqual(checked([binary]), '')
        for name, command in [('c', [ROOT / 'bin/nano_virt']), ('nanolang', source_compiler_command())]:
            if not command:
                continue
            module = self.directory / ('imported-' + name + '.nvm')
            checked([*command, self.fixture, '--test-imports', '--emit-nvm', '-o', module], build=True)
            checked([ROOT / 'bin/nano_vm', '--verify-only', module])
            self.assertEqual(checked([ROOT / 'bin/nano_vm', module, '--']), '')


if __name__ == '__main__':
    unittest.main()
