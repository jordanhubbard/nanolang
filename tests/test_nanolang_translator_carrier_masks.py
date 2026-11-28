"""I compare every physical scalar-carrier input without granting admission."""
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
    for op, value in rows:
        result.append(op)
        if op == 1:
            result.extend(struct.pack('<H', value))
        else:
            result.append(value)
    return bytes(result)


def complete_packets():
    groups = [('kinds', [(0, k) for k in range(256)]),
              ('masks', [(1, k) for k in range(65536)])]
    return [(f'{name}-{i // 4096:02d}', packet(rows[i:i + 4096]))
            for name, rows in groups for i in range(0, len(rows), 4096)]


class CarrierMasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='nano-carrier-masks-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory = Path(cls.workspace.name)
        source = (ROOT / 'src/nanoisa/nvm2c.c').read_bytes()
        markers = [b'#define NVM2C_VK_INT 0', b'static int variant_field(',
                   b'static int variant_scalar_tags(', b'static int variant_scalar_kind(',
                   b'#define NVM2C_SCALAR_UNKNOWN', b'#include "nvm2c_effect_flow.h"']
        assert all(source.count(marker) == 1 for marker in markers)
        indices = [source.index(marker) for marker in markers]
        actual = b''.join(source[a:b] for a, b in zip(indices[::2], indices[1::2]))
        assert actual.count(b'static ') == 5
        (cls.directory / 'nvm2c_carrier_actual.h').write_bytes(actual)
        (cls.directory / 'reference-source.json').write_text(json.dumps({
            'source': 'src/nanoisa/nvm2c.c', 'source_sha256': hashlib.sha256(source).hexdigest(),
            'verbatim_helper_sha256': hashlib.sha256(actual).hexdigest(),
            'markers': [m.decode() for m in markers], 'functions': 5,
            'product_source_edited': False}, indent=2) + '\n')
        cls.references = []
        for name, flags in [('ordinary', ['-O2']),
                            ('sanitized', ['-O1', '-g', '-fsanitize=address,undefined',
                                           '-fno-sanitize-recover=all'])]:
            binary = cls.directory / ('reference-' + name)
            checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-Wall', '-Wextra',
                     '-Werror', *flags, '-I', cls.directory,
                     ROOT / 'tests/nvm2c_carrier_masks_reference.c',
                     *native_link_flags(), '-o', binary], build=True)
            cls.references.append((name, [binary]))
        cls.fixture = ROOT / 'tests/nvm2c_carrier_masks_driver.nano'
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
        kinds = {int(r[1]): int(r[2]) for r in parsed if r[0] == 'K'}
        masks = {int(r[1]): tuple(map(int, r[2:])) for r in parsed if r[0] == 'T'}
        self.assertEqual((len(kinds), len(masks)), (256, 65536))
        self.assertEqual(sum(int(r[1]) for r in parsed if r[0] == 'Z'), 65792)
        known = {0: 2, 1: 32, 9: 16, 11: 8, 13: 2048}
        self.assertEqual(kinds, {k: known.get(k, 32768) for k in range(256)})
        members = [2, 4, 8, 16, 32, 128, 512]
        marked = {16384 + sum(bit for i, bit in enumerate(members) if subset & (1 << i)) + void
                  for subset in range(1, 128) for void in (0, 1)}
        boxed = marked | {6, 7, 10, 11}
        optional = {3, 5, 9, 17}
        self.assertEqual((len(marked), len(boxed)), (254, 258))
        self.assertEqual({mask for mask, facts in masks.items() if facts[0]}, boxed)
        self.assertEqual({mask for mask, facts in masks.items() if facts[1]}, optional)
        self.assertFalse(boxed & optional)
        for mask, expected in [(0, (0, 0)), (1, (0, 0)), (2, (0, 0)), (3, (0, 1)),
                               (6, (1, 0)), (7, (1, 0)), (10, (1, 0)), (11, (1, 0)),
                               (17, (0, 1)), (33, (0, 0)), (2049, (0, 0)),
                               (16384, (0, 0)), (16385, (0, 0)), (16386, (1, 0)),
                               (16512, (1, 0)), (32768, (0, 0)), (65535, (0, 0))]:
            self.assertEqual(masks[mask], expected)

    def test_malformed_and_prior_packet_boundaries(self):
        sample = packet([(0, 255), (1, 65535)])
        cases = [(f'truncated-{n}', sample[:n]) for n in range(len(sample))]
        cases += [('unknown-op', struct.pack('<I', 1) + bytes([2, 0])),
                  ('late-unknown', struct.pack('<I', 2) + bytes([0, 13, 2, 0])),
                  ('huge-count', struct.pack('<I', 0xffffffff)),
                  ('count-cap', struct.pack('<I', 262145)), ('trailing', sample + b'\0'),
                  ('size-cap', b'\0' * 1048577)]
        self.assertEqual(self.compare(cases, 'malformed'), 'P\n' * len(cases))
        neighbors = [('before', packet([(0, 13)])), ('late', struct.pack('<I', 2) + bytes([0, 13, 2, 0])),
                     ('after', packet([(1, 17)])), ('empty', packet([]))]
        self.assertEqual(self.compare(neighbors, 'neighbors'), 'Z|1\nK|13|2048\nP\nZ|1\nT|17|0|1\nZ|0\n')

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
