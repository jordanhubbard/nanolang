"""I compare every physical classifier fact input without granting admission."""
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
    return struct.pack('<I', len(rows)) + b''.join(rows)


def cell(policy, current, incoming, flags=0):
    return bytes((0, policy, current, incoming, flags))


def custom(policy, current, incoming, flags, role, ordinal, function=17, offset=29, local=3, record=4):
    return struct.pack('<6B5I', 2, policy, current, incoming, flags, role, ordinal, function, offset, local, record)


def vector(policy, current, incoming, flags=0, role=2, ordinal=0):
    assert len(current) == len(incoming)
    return struct.pack('<4BIH', 3, policy, flags, role, ordinal, len(current)) + bytes(current) + bytes(incoming)


def copy_case(width, fill, source=None, flags=0):
    assert source is None or len(source) == width
    return struct.pack('<4BH', 4, flags, fill, int(source is not None), width) + (b'' if source is None else bytes(source))


def sequence(start, steps, flags=0):
    return struct.pack('<3BH', 5, flags, start, len(steps)) + bytes(v for row in steps for v in row)


def grouped(rows, prefix='rows'):
    return [(f'{prefix}-{i // 4096:02d}', packet(rows[i:i + 4096])) for i in range(0, len(rows), 4096)]


def local_type(kind):
    return {1: 'const char *', 3: 'narr_t', 4: 'nrec_t', 5: 'nsarr_t', 6: 'nrarr_t',
            7: 'nmap_t', 8: 'nmap_value', 9: 'bool', 10: 'narr_t', 11: 'double',
            12: 'narr_t', 13: 'nmap_value', 14: 'narr_t', 15: 'narr_t',
            16: 'narr_t', 17: 'narr_t'}.get(kind, 'int64_t')


def target(role, ordinal, local, record):
    if role == 1: return f'parameter {ordinal % local} of function {ordinal // local}'
    if role == 2:
        parameter, field = divmod(ordinal, record)
        return f'field {field} of parameter {parameter % local} of function {parameter // local}'
    if role == 3: return f'result field {ordinal % record} of function {ordinal // record}'
    if role == 4: return f'array result of function {ordinal}'
    return 'effect representation'


def expected_cell(policy, current, incoming, changed=False, failed=False, error='',
                  role=None, ordinal=None, function=17, offset=29, local=3, record=4, field=0):
    if role is None: role, ordinal = ((0, 1) if policy >= 4 else (3, 4) if policy == 3 else (1, 5))
    scalar = {0, 1, 9, 11}; containers = {3, 5, 7, 10, 12, 14, 15, 16, 17}
    admitted = current == incoming or incoming == 2 or current == 2
    result = incoming if current == 2 and incoming != 2 else current
    if policy in (1, 2, 3) and ((current == 8 and incoming in scalar) or (incoming == 8 and current in scalar)):
        admitted, result = True, 8
    if policy == 2:
        widen = {(8, k): k for k in (4, 13)} | {(k, 8): k for k in (4, 13)}
        widen |= {(8, k): 8 for k in containers} | {(k, 8): 8 for k in containers}
        if (current, incoming) in widen: admitted, result = True, widen[current, incoming]
    if policy == 3:
        finite = set(range(65, 128))
        if current in finite and incoming in finite: admitted, result = True, current | incoming
        elif current in finite and incoming == 8: admitted, result = True, current | 16
        elif current == 8 and incoming in finite: admitted, result = True, incoming | 16
    if policy == 4:
        if incoming in (2, 8) or current == incoming: admitted, result = True, current
        elif current == 2: admitted, result = True, incoming
        elif 6 in (current, incoming): admitted, result = False, current
        else: admitted, result = True, 8
    if policy == 6:
        admitted = True
        result = current if incoming == 2 or current in (incoming, 8) else incoming if current == 2 else 8
    if not admitted:
        if not failed:
            site = f'function {function}, offset {offset}'
            if policy == 4: error = f'I require record-array global {ordinal} to retain one exact representation ({site})'
            elif policy == 5: error = f'I found conflicting field {field} representations in record-array global {ordinal} ({site})'
            else: error = f'I cannot assign conflicting kinds to a function parameter or aggregate field ({site}: {local_type(current)} versus {local_type(incoming)} at {target(role, ordinal, local, record)})'
        failed = True
    return (1, int(admitted), result, int(changed or result != current), int(failed), error)


def flags_state(flags): return (bool(flags & 1), bool(flags & 2), 'prior' if flags & 4 else '')


class RepresentationFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='nano-representation-facts-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory = Path(cls.workspace.name)
        source = (ROOT / 'src/nanoisa/nvm2c.c').read_bytes()
        declaration_begin = b'typedef struct Nvm2cFieldBlock {'
        declaration_end = b'typedef struct Nvm2cJoinShape {'
        declarations = source[source.index(declaration_begin):source.index(declaration_end)]
        (cls.directory / 'nvm2c_representation_declarations.h').write_bytes(declarations)
        markers = [b'#define NVM2C_VK_INT 0', b'static int variant_field(',
                   b'static void nvm2c_fail(', b'static int nvm2c_grow(',
                   b'static int variant_field(', b'static uint8_t variant_field_for(',
                   b'static int word_array_storage(', b'static int sortable_array_storage(',
                   b'static const char *c_local_type(', b'static uint8_t fn_local_kind(',
                   b'typedef struct {\n    uint8_t *parameters;', b'static int shape_ok(']
        assert all(source.count(marker) == 1 for marker in set(markers))
        indices = [source.index(marker) for marker in markers]
        actual = b''.join(source[a:b] for a, b in zip(indices[::2], indices[1::2]))
        assert actual.count(b'static ') == 14
        (cls.directory / 'nvm2c_representation_actual.h').write_bytes(actual)
        (cls.directory / 'reference-source.json').write_text(json.dumps({
            'source': 'src/nanoisa/nvm2c.c', 'source_sha256': hashlib.sha256(source).hexdigest(),
            'verbatim_helper_sha256': hashlib.sha256(actual).hexdigest(),
            'declarations_sha256': hashlib.sha256(declarations).hexdigest(),
            'markers': [m.decode() for m in markers], 'functions': 14,
            'product_source_edited': False}, indent=2) + '\n')
        cls.references = []
        for name, flags in [('ordinary', ['-O2']),
                            ('sanitized', ['-O1', '-g', '-fsanitize=address,undefined',
                                           '-fno-sanitize-recover=all'])]:
            binary = cls.directory / ('reference-' + name)
            checked([*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-Wall', '-Wextra',
                     '-Werror', *flags, '-I', cls.directory,
                     ROOT / 'tests/nvm2c_representation_facts_reference.c',
                     *native_link_flags(), '-o', binary], build=True)
            cls.references.append((name, [binary]))
        cls.fixture = ROOT / 'tests/nvm2c_representation_facts_driver.nano'
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

    def test_every_kind_pair_for_every_policy(self):
        for policy in range(7):
            with self.subTest(policy=policy):
                result = self.compare(grouped([cell(policy, a, b) for a in range(256) for b in range(256)]), f'policy-{policy}')
                rows = [r.split('|') for r in result.splitlines() if r.startswith('C|')]
                self.assertEqual(len(rows), 65536)
                for row, (a, b) in zip(rows, ((a, b) for a in range(256) for b in range(256))):
                    self.assertEqual(list(map(int, row[1:5])), [policy, 0, a, b])
                    self.assertEqual((*map(int, row[5:10]), row[10]), expected_cell(policy, a, b), (policy, a, b))

    def test_exact_global_map_pair_domains(self):
        tuples = [(s, k, 0, 1) for s in range(256) for k in range(256)]
        tuples += [(1, 7, a, b) for a in range(256) for b in range(256) if (a, b) != (0, 1)]
        result = self.compare(grouped([bytes((1, *t)) for t in tuples]), 'map-pairs')
        rows = [list(map(int, r.split('|')[1:])) for r in result.splitlines() if r.startswith('M|')]
        self.assertEqual(len(rows), 131071)
        for row, t in zip(rows, tuples):
            self.assertEqual(row[:4], list(t))
            self.assertEqual(row[4], int(t[0] == 1 and t[1] == 7 and t[2] in (0, 1) and t[3] in (0, 1)))

    def test_sticky_state_locations_vectors_and_copies(self):
        kinds = [*range(18), 65, 80, 96, 127, 128, 255]
        rows = []; expected = []
        for flags in range(8):
            for policy in range(7):
                for a in kinds:
                    for b in kinds:
                        rows.append(cell(policy, a, b, flags)); expected.append(expected_cell(policy, a, b, *flags_state(flags)))
        locations = [(0, 0, 3, 4), (1, 5, 3, 4), (2, 23, 3, 4), (3, 7, 3, 4), (4, 1, 3, 4),
                     (1, 133117, 66559, 1), (2, 131069, 1, 65535), (3, 131069, 1, 65535)]
        for role, ordinal, local, record in locations:
            for function, offset in [(0, 0), (4294967295, 4294967295)]:
                rows.append(custom(0, 0, 9, 0, role, ordinal, function, offset, local, record))
                expected.append(expected_cell(0, 0, 9, role=role, ordinal=ordinal, function=function, offset=offset, local=local, record=record))
        result = self.compare(grouped(rows), 'state-locations')
        observed = [r.split('|') for r in result.splitlines() if r.startswith('C|')]
        self.assertEqual(len(observed), len(expected))
        for row, want in zip(observed, expected): self.assertEqual((*map(int, row[5:10]), row[10]), want)
        vectors = []
        for flags in range(8):
            for policy in (0, 3, 5):
                for width in (1, 4, 80):
                    for conflict in range(width):
                        a = [2] * width; b = [7] * width; a[conflict] = 0; b[conflict] = 9
                        role, ordinal = (0, 1) if policy == 5 else (3, 0) if policy == 3 else (2, 0)
                        vectors.append((policy, flags, role, ordinal, a, b))
            vectors.append((3, flags, 3, 0, [65, 80, 8, 2], [96, 8, 65, 11]))
            vectors.append((6, flags, 0, 1, [0, 2, 255, 9], [1, 1, 0, 0]))
        result = self.compare(grouped([vector(p, a, b, f, role, ordinal) for p, f, role, ordinal, a, b in vectors]), 'vectors')
        observed = [r.split('|') for r in result.splitlines() if r.startswith('V|')]
        self.assertEqual(len(observed), len(vectors))
        for row, (policy, flags, role, ordinal, a, b) in zip(observed, vectors):
            values = list(a); state = flags_state(flags); accepted = 1
            for i in range(min(2, len(a)) if policy == 6 else len(a)):
                result = expected_cell(policy, values[i], b[i], *state, role=role,
                                       ordinal=ordinal if policy >= 5 else ordinal + i, record=len(a), field=i)
                values[i] = result[2]; state = (bool(result[3]), bool(result[4]), result[5]); accepted = result[1]
                if not accepted: break
            self.assertEqual(list(map(int, row[1:7])), [policy, flags, role, ordinal, 1, accepted])
            self.assertEqual((*map(int, row[7:9]), row[9]), (int(state[0]), int(state[1]), state[2]))
            self.assertEqual(list(map(int, row[10].split(','))), values)
            self.assertEqual(list(map(int, row[11].split(','))), a)
        copies = [(4, fill, None, 0) for fill in range(256)]
        copies += [(n, 255, bytes(i % 256 for i in range(n)), flags) for n in (0, 1, 4, 80, 256, 1024, 65535) for flags in range(8)]
        result = self.compare(grouped([copy_case(n, fill, src, flags) for n, fill, src, flags in copies]), 'copies')
        observed = [r.split('|') for r in result.splitlines() if r.startswith('D|')]
        self.assertEqual(len(observed), len(copies))
        for row, (n, fill, source, flags) in zip(observed, copies):
            want = list(source) if source is not None else [fill] * n
            if source is not None and n: want[0] = (want[0] + 1) % 256
            self.assertEqual(list(map(int, row[1:10])), [flags, fill, int(source is not None), n, 1, 1, 1, flags & 1, (flags >> 1) & 1])
            self.assertEqual(row[10], flags_state(flags)[2])
            self.assertEqual(list(map(int, row[11].split(','))) if row[11] else [], want)

    def test_continued_merges_preserve_first_failure(self):
        starts = [*range(18), 65, 80, 96, 127, 255]
        patterns = [[(0, 9), (0, 2), (1, 8), (2, 13), (4, 6), (6, 2), (6, 1)],
                    [(3, 65), (3, 96), (3, 8), (3, 2), (0, 11), (2, 4)],
                    [(4, 7), (4, 8), (4, 6), (0, 2), (6, 1)], []]
        cases = [(flags, start, steps) for flags in range(8) for start in starts for steps in patterns]
        result = self.compare(grouped([sequence(start, steps, flags) for flags, start, steps in cases]), 'sequences')
        rows = [r.split('|') for r in result.splitlines() if r.startswith('Q|')]
        self.assertEqual(len(rows), len(cases))
        for row, (flags, kind, steps) in zip(rows, cases):
            self.assertEqual(list(map(int, row[1:4])), [flags, kind, len(steps)])
            observed = row[4].split('~') if steps else []
            state = flags_state(flags)
            self.assertEqual(len(observed), len(steps))
            for text, (policy, incoming) in zip(observed, steps):
                want = expected_cell(policy, kind, incoming, *state);kind = want[2];state=(bool(want[3]),bool(want[4]),want[5]);parts=text.split('^')
                self.assertEqual((*map(int, parts[:6]), parts[6]), (policy,incoming,want[1],kind,want[3],want[4],want[5]))

    def test_malformed_and_prior_packet_boundaries(self):
        sample = packet([cell(0, 2, 7), bytes((1, 1, 7, 0, 1)), sequence(0, [(0, 9), (1, 8)])])
        cases = [(f'truncated-{n}', sample[:n]) for n in range(len(sample))]
        cases += [('unknown-op', packet([bytes((6,0,0,0,0))])), ('late-unknown', packet([cell(0,2,7),bytes((6,0,0,0,0))])),
                  ('policy', packet([cell(7,0,0)])), ('flags', packet([cell(0,0,0,8)])),
                  ('context-zero-width', packet([custom(0,0,9,0,0,0,local=0)])),
                  ('context-huge', packet([custom(0,0,9,0,0,0,local=66559,record=65535)])),
                  ('context-bad-slot', packet([custom(0,0,9,0,1,6)])),
                  ('global-bad-slot', packet([custom(4,0,9,0,0,2)])),
                  ('vector-policy', packet([vector(2,[0],[1])])),
                  ('map-short', packet([vector(6,[0],[1],role=0,ordinal=1)])),
                  ('huge-count',struct.pack('<I',0xffffffff)),('trailing',sample+b'\0'),('size-cap',b'\0'*1048577)]
        self.assertEqual(self.compare(cases,'malformed'),'P\n'*len(cases))
        neighbors=[('before',packet([cell(0,2,7)])),('refused',packet([cell(0,2,7),bytes((6,0,0,0,0))])),('after',packet([bytes((1,1,7,0,1))])),('empty',packet([]))]
        self.assertEqual(self.compare(neighbors,'neighbors'),'Z|1\nC|0|0|2|7|1|1|7|1|0|\nP\nZ|1\nM|1|7|0|1|1\nZ|0\n')

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
