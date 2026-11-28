"""I compare every physical C ABI type input without granting admission."""
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
    for op, count, tag, kind in rows:
        result.extend((op, count))
        if op == 0:
            result.extend((tag, kind))
    return bytes(result)


def grouped_packets(groups):
    return [(f'{name}-{i // 4096:02d}', packet(rows[i:i + 4096]))
            for name, rows in groups for i in range(0, len(rows), 4096)]


def complete_packets():
    return grouped_packets([
        ('count-tag', [(0, c, t, 2) for c in range(256) for t in range(256)]),
        ('count-array', [(0, c, 7, k) for c in range(256) for k in range(256) if k != 2]),
        ('locals', [(1, k, 0, 0) for k in range(256)]),
        ('aggregates', [(2, t, 0, 0) for t in range(256)])])


def expected_result(count, tag, kind):
    if count == 0 or tag == 0:
        text = 'void'
    elif count != 1:
        text = ''
    elif tag == 7:
        text = ('narr_t' if kind in {3, 10, 12, 14, 15, 16, 17}
                else {5: 'nsarr_t', 6: 'nrarr_t'}.get(kind, ''))
    else:
        text = {1: 'int64_t', 2: 'nmap_value', 3: 'double', 4: 'int64_t',
                5: 'const char *', 6: 'nmap_value', 8: 'nrec_t', 9: 'nmap_value',
                10: 'nrec_t', 11: 'nmap_value', 12: 'nrec_t', 13: 'nmap_t',
                14: 'nmap_value', 15: 'nmap_value'}.get(tag, '')
    return (int(bool(text)), int(count == 1 and tag in {1, 4}), text)


class CAbiTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='nano-c-abi-types-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory = Path(cls.workspace.name)
        source = (ROOT / 'src/nanoisa/nvm2c.c').read_bytes()
        markers = [b'#define NVM2C_VK_INT 0', b'static int variant_field(',
                   b'static int word_array_storage(', b'static int sortable_array_storage(',
                   b'static int aggregate_value_tag(', b'static uint8_t fn_local_kind(']
        assert all(source.count(marker) == 1 for marker in markers)
        indices = [source.index(marker) for marker in markers]
        actual = b''.join(source[a:b] for a, b in zip(indices[::2], indices[1::2]))
        assert actual.count(b'static ') == 6
        (cls.directory / 'nvm2c_c_abi_actual.h').write_bytes(actual)
        (cls.directory / 'reference-source.json').write_text(json.dumps({
            'source': 'src/nanoisa/nvm2c.c', 'source_sha256': hashlib.sha256(source).hexdigest(),
            'verbatim_helper_sha256': hashlib.sha256(actual).hexdigest(),
            'markers': [m.decode() for m in markers], 'functions': 6,
            'product_source_edited': False}, indent=2) + '\n')
        cls.references = []
        for name, flags in [('ordinary', ['-O2']),
                            ('sanitized', ['-O1', '-g', '-fsanitize=address,undefined',
                                           '-fno-sanitize-recover=all'])]:
            binary = cls.directory / ('reference-' + name)
            checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-Wall', '-Wextra',
                     '-Werror', *flags, '-I', cls.directory,
                     ROOT / 'tests/nvm2c_c_abi_types_reference.c',
                     *native_link_flags(), '-o', binary], build=True)
            cls.references.append((name, [binary]))
        cls.fixture = ROOT / 'tests/nvm2c_c_abi_types_driver.nano'
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

    def test_complete_branch_domains(self):
        result = self.compare(complete_packets(), 'complete')
        parsed = [line.split('|') for line in result.splitlines()]
        results = {(int(r[1]), int(r[2]), int(r[3])): (int(r[4]), int(r[5]), r[6])
                   for r in parsed if r[0] == 'R'}
        locals_ = {int(r[1]): r[2] for r in parsed if r[0] == 'L'}
        aggregates = {int(r[1]): (int(r[2]), int(r[3])) for r in parsed if r[0] == 'A'}
        self.assertEqual((len(results), len(locals_), len(aggregates)), (130816, 256, 256))
        self.assertEqual(sum(int(r[1]) for r in parsed if r[0] == 'Z'), 131328)
        expected_keys = {(c, t, 2) for c in range(256) for t in range(256)}
        expected_keys |= {(c, 7, k) for c in range(256) for k in range(256)}
        self.assertEqual(set(results), expected_keys)
        for key, facts in results.items():
            self.assertEqual(facts, expected_result(*key), key)
        known = {11: 'double', 1: 'const char *', 5: 'nsarr_t', 4: 'nrec_t',
                 6: 'nrarr_t', 7: 'nmap_t', 8: 'nmap_value', 13: 'nmap_value'}
        known.update({k: 'narr_t' for k in {3, 10, 12, 14, 15, 16, 17}})
        self.assertEqual(locals_, {k: known.get(k, 'int64_t') for k in range(256)})
        self.assertEqual(aggregates, {t: (int(t in {8, 10, 12}), {8: 0, 10: 1}.get(t, 2))
                                      for t in range(256)})
        self.assertEqual(results[255, 0, 2], (1, 0, 'void'))
        self.assertEqual(results[2, 1, 2], (0, 0, ''))
        self.assertEqual(results[1, 7, 255], (0, 0, ''))
        self.assertEqual(results[1, 7, 5], (1, 0, 'nsarr_t'))
        self.assertEqual(results[1, 7, 6], (1, 0, 'nrarr_t'))

    def test_ignored_array_context(self):
        rows = [(0, c, t, k) for c in (0, 1, 2, 255) for t in range(256)
                for k in (0, 2, 3, 5, 6, 255)]
        result = self.compare(grouped_packets([('contexts', rows)]), 'contexts')
        groups = {}
        for line in result.splitlines():
            r = line.split('|')
            if r[0] != 'R':
                continue
            count, tag, kind = map(int, r[1:4])
            facts = (int(r[4]), int(r[5]), r[6])
            self.assertEqual(facts, expected_result(count, tag, kind))
            groups.setdefault((count, tag), set()).add(facts)
        self.assertEqual(len(groups), 1024)
        for key, facts in groups.items():
            self.assertEqual(len(facts), 4 if key == (1, 7) else 1, key)

    def test_malformed_and_prior_packet_boundaries(self):
        sample = packet([(0, 1, 7, 3), (1, 255, 0, 0), (2, 10, 0, 0)])
        cases = [(f'truncated-{n}', sample[:n]) for n in range(len(sample))]
        cases += [('unknown-op', struct.pack('<I', 1) + bytes([3, 0])),
                  ('late-unknown', struct.pack('<I', 2) + bytes([1, 255, 3, 0])),
                  ('huge-count', struct.pack('<I', 0xffffffff)),
                  ('count-cap', struct.pack('<I', 262145)), ('trailing', sample + b'\0'),
                  ('size-cap', b'\0' * 1048577)]
        self.assertEqual(self.compare(cases, 'malformed'), 'P\n' * len(cases))
        neighbors = [('before', packet([(0, 1, 1, 2)])), ('late', struct.pack('<I', 2) + bytes([1, 255, 3, 0])),
                     ('after', packet([(1, 255, 0, 0)])), ('empty', packet([]))]
        self.assertEqual(self.compare(neighbors, 'neighbors'), 'Z|1\nR|1|1|2|1|1|int64_t\nP\nZ|1\nL|255|int64_t\nZ|0\n')

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
