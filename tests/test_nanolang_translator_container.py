"""I compare my NanoLang v2 container boundary with the C reference."""
from pathlib import Path
import os
import random
import shlex
import struct
import sys
import tempfile
import unittest
import zlib

from tests.native_process import native_link_flags, run_captured

ROOT = Path(__file__).resolve().parents[1]


def container(sections=(), *, features=0, isa=2):
    """I construct wire bytes independently of either decoder."""
    floor = 40 + len(sections) * 24
    directory, payload = bytearray(), bytearray()
    for kind, data in sections:
        directory += struct.pack('<IIQQ', kind, 0, floor + len(payload), len(data))
        payload += data
    body = directory + payload
    return struct.pack('<4sHHIQIIIII', b'NVM\x02', 1, isa, features,
                       40 + len(body), 40, len(sections), 0xffffffff, 0,
                       zlib.crc32(body)) + body


def patched(data, offset, encoding, value):
    result = bytearray(data)
    struct.pack_into(encoding, result, offset, value)
    return bytes(result)


def corpus():
    cases = [('empty-container', container(), 0),
             ('all-known-features', container(features=1023), 0),
             # ISA and feature/payload consistency belong to module admission.
             ('isa-is-not-container-admission', container(isa=65535), 0),
             ('all-section-kinds', container([(i, bytes([i, 0, 255])) for i in range(1, 15)]), 0),
             ('unsorted-directory', container([(6, b'abc'), (1, b'def')]), 0),
             ('empty-sections', container([(1, b''), (14, b'')]), 0)]
    base = container([(1, b'abc'), (6, b'\0\xffdef')])
    cases.extend((f'truncated-{size}', base[:size], 1) for size in range(40))
    mutations = [
        ('old-lineage', 3, '<B', 1, 2), ('format', 4, '<H', 2, 3),
        ('header-size', 20, '<I', 41, 4), ('feature', 8, '<I', 1024, 5),
        ('huge-total', 12, '<Q', 2**64-1, 6),
        ('negative-host-total', 12, '<Q', 2**63, 6),
        ('reserved-header', 32, '<I', 1, 7),
        ('directory-count', 24, '<I', 2**32-1, 1),
        ('section-type-zero', 40, '<I', 0, 8),
        ('section-type-high', 40, '<I', 15, 8),
        ('reserved-section', 44, '<I', 1, 7),
        ('duplicate', 64, '<I', 1, 11),
        ('huge-offset', 48, '<Q', 2**64-1, 9),
        ('signed-offset-boundary', 48, '<Q', 2**63, 9),
        ('huge-size', 56, '<Q', 2**64-1, 9),
        ('wrap-range', 56, '<Q', 2**64-88, 9),
        ('header-overlap', 48, '<Q', 0, 10),
        ('directory-overlap', 48, '<Q', 40, 10),
        ('payload-overlap', 72, '<Q', 89, 10),
        ('bad-checksum', 36, '<I', 0, 12),
    ]
    cases.extend((name, patched(base, offset, encoding, value), expected)
                 for name, offset, encoding, value, expected in mutations)
    # I compare precedence when both structure and checksum are broken.
    duplicate = patched(base, 64, '<I', 1)
    cases.append(('duplicate-before-range', patched(duplicate, 72, '<Q', 2**64-1), 11))
    empty = container([(1, b'')])
    moved = patched(empty, 48, '<Q', 0)
    moved = patched(moved, 36, '<I', zlib.crc32(moved[40:]))
    cases.append(('empty-range-may-touch-header', moved, 0))
    cases.append(('empty-range-must-fit-file', patched(empty, 48, '<Q', 65), 9))
    randomizer = random.Random(1142)
    for i in range(48):
        payload = bytes(randomizer.randrange(256) for _ in range(i))
        data = container([(1, payload), (6, payload[::-1])])
        cases.append((f'crc-payload-{i}', data, 0))
        # I mutate any byte, including fields whose semantics this layer leaves
        # to the payload decoder. The C reference supplies the expected status.
        changed = bytearray(data)
        index = randomizer.randrange(len(data))
        changed[index] ^= 1 << randomizer.randrange(8)
        cases.append((f'wire-mutation-{i}', bytes(changed), None))
    return cases


class NanoLangTranslatorContainer(unittest.TestCase):
    @classmethod
    def checked(cls, command, *, build=False, timeout=60):
        completed = run_captured(command, build=build, timeout=timeout, cwd=ROOT)
        if completed.returncode:
            raise AssertionError(f'{command}: {completed.returncode}\n'
                                 + completed.stdout.decode(errors='replace')
                                 + completed.stderr.decode(errors='replace'))
        return completed.stdout.decode()

    def test_real_wire_decisions_in_native_vm_and_aot(self):
        with tempfile.TemporaryDirectory(prefix='nano-nanolang-container-') as directory:
            directory = Path(directory)
            fixture = ROOT / 'tests/nvm2c_container_driver.nano'
            native = directory / 'container-native'
            module = directory / 'container.nvm'
            source = directory / 'container.c'
            aot = directory / 'container-aot'
            self.checked([ROOT / 'bin/nanoc_c', fixture, '-o', native], build=True)
            self.checked([ROOT / 'bin/nano_virt', fixture, '-o', module], build=True)
            compiler = shlex.split(os.environ.get('CC', 'cc'))

            def native_from_module(input_module, c_path, executable):
                self.checked([ROOT / 'bin/nvm2c', input_module, '-o', c_path], build=True)
                self.checked([*compiler, '-std=c11', '-O1', '-g',
                              '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                              c_path, ROOT / 'bin/nano_aot_runtime.o', '-lm',
                              *(['-ldl', '-Wl,--export-dynamic'] if sys.platform.startswith('linux') else []),
                              *native_link_flags(), '-o', executable], build=True)

            native_from_module(module, source, aot)
            routes = [('c-seed-native', [native]),
                      ('c-producer-nanoisa-vm', [ROOT / 'bin/nano_vm', module, '--']),
                      ('c-producer-sanitized-native', [aot])]
            source_compiler = os.environ.get('NANOLANG_TRANSLATOR_SOURCE_COMPILER')
            if source_compiler:
                source_module = directory / 'source-container.nvm'
                source_c = directory / 'source-container.c'
                source_native = directory / 'source-container-aot'
                self.checked([source_compiler, fixture, '--emit-nvm', '-o', source_module], build=True)
                native_from_module(source_module, source_c, source_native)
                routes.extend([
                    ('nanolang-producer-nanoisa-vm', [ROOT / 'bin/nano_vm', source_module, '--']),
                    ('nanolang-producer-sanitized-native', [source_native])])
            cases = corpus()
            paths = []
            for index, (_, data, _) in enumerate(cases):
                path = directory / f'{index}.nvm'
                path.write_bytes(data)
                paths.append(path)
            reference = list(map(int, self.checked([ROOT / 'obj/nvm2c_container_reference', *paths]).splitlines()))
            self.assertEqual(len(reference), len(cases))
            for (name, _, expected), actual in zip(cases, reference):
                if expected is not None:
                    with self.subTest(reference_contract=name):
                        self.assertEqual(actual, expected)
            for name, command in routes:
                output = list(map(int, self.checked([*command, *paths]).splitlines()))
                self.assertEqual(len(output), len(cases), name)
                for (case, _, _), expected, actual in zip(cases, reference, output):
                    with self.subTest(route=name, case=case):
                        self.assertEqual(actual, expected)
            print(f'I matched {len(cases)} real container decisions across {len(routes)} execution routes.')
