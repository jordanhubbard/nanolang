"""I compare complete function boundaries and resolved targets with the VM."""
from pathlib import Path
import json
import random
import struct
import tempfile
import unittest

from tests.nanolang_translator_routes import build_routes, checked

ROOT = Path(__file__).resolve().parents[1]


def carrier(code, start=0, length=None, functions=1):
    if length is None:
        length = len(code) - start
    return struct.pack('<QQQ', start, length, functions) + code


def corpus(widths):
    cases = [('empty', carrier(b''), True), ('no-functions', carrier(b'', functions=0), False),
             ('nop', carrier(b'\0'), True), ('invalid', carrier(b'\xff'), False),
             ('outside-function-is-not-decoded', carrier(b'\xff\0\xff', 1, 1), True)]
    for size in range(24):
        cases.append((f'short-header-{size}', bytes(size), False))
    for field in ('start', 'length', 'functions'):
        for value in (2**32, 2**63 - 1, 2**63, 2**64 - 1):
            kwargs = {field: value}
            if field == 'start':
                kwargs['length'] = 0
            cases.append((f'wide-{field}-{value}', carrier(b'\0', **kwargs), False))
    for start, length, valid in ((0, 0, True), (1, 0, True), (2, 0, False),
                                  (0, 2, False), (1, 1, False)):
        cases.append((f'window-{start}-{length}', carrier(b'\0', start, length), valid))
    for opcode, width in enumerate(widths):
        if not width:
            cases.append((f'unknown-{opcode}', carrier(bytes([opcode])), False))
            continue
        instruction = bytes([opcode]) + bytes(width - 1)
        # Every branch targets its own start; File results have a following arm.
        cases.append((f'opcode-{opcode}', carrier(instruction + bytes(2), functions=2), True))
        cases.append((f'offset-opcode-{opcode}', carrier(b'\xff' + instruction + bytes(2) + b'\xff',
                                                      1, width + 2, 2), True))
        for size in range(1, width):
            cases.append((f'truncated-{opcode}-{size}', carrier(instruction[:size]), False))
    for opcode, prefix, suffix in ((56, b'', b''), (57, b'', b''), (58, b'', b''),
                                    (107, bytes(2), b''), (165, bytes(4), bytes(4)),
                                    (146, bytes(2), b'')):
        width = 1 + len(prefix) + 4 + len(suffix)
        boundaries = {0, 1, width + 1, width + 2, width + 3}
        for relative in (*range(-20, 41), -(2**31), 2**31 - 1):
            instruction = bytes([opcode]) + prefix + struct.pack('<i', relative) + suffix
            target = 1 + relative
            valid = target in boundaries and (opcode != 146 or target < width + 3)
            cases.append((f'branch-{opcode}-{relative}', carrier(b'\xff'*3 + b'\0' + instruction + bytes(2),
                                                               3, width + 3), valid))
        instruction = bytes([opcode]) + prefix + struct.pack('<i', width) + suffix
        cases.append((f'terminal-{opcode}', carrier(instruction), opcode != 146))
    for opcode in (51, 59):
        for functions in range(4):
            for target in (0, 1, 2, 3, 2**31, 2**32 - 1):
                cases.append((f'call-{opcode}-{functions}-{target}',
                              carrier(bytes([opcode]) + struct.pack('<I', target), functions=functions),
                              functions > 0 and target < functions))
    cases.append(('unreachable-invalid-instruction', carrier(b'\x38\x06\0\0\0\xff'), False))
    rng = random.Random(1142006)
    supported = [op for op, width in enumerate(widths) if width]
    for index in range(128):
        code = b''.join(bytes([op]) + rng.randbytes(widths[op] - 1)
                        for op in rng.choices(supported, k=rng.randrange(1, 25)))
        cases.append((f'random-{index}', carrier(code, functions=3), None))
    return cases


class NanoLangTranslatorFunctions(unittest.TestCase):
    def test_complete_function_decode(self):
        with tempfile.TemporaryDirectory(prefix='nano-nanolang-functions-') as name:
            work = Path(name)
            # I obtain only the primary widths from the independent C ISA.
            probes = []
            for opcode in range(256):
                path = work / f'width-{opcode}.bin'
                path.write_bytes(struct.pack('<QQ', 0, 64) + bytes([opcode]) + bytes(63))
                probes.append(path)
            widths = []
            for start in range(0, 256, 64):
                reports = checked([ROOT / 'obj/nvm2c_instruction_reference', *probes[start:start + 64]])
                widths.extend(int(row.split(':')[0]) for row in reports.splitlines())
            self.assertEqual(len(widths), 256)
            cases = corpus(widths)
            paths = []
            for i, (_, data, _) in enumerate(cases):
                path = work / f'{i}.bin'; path.write_bytes(data); paths.append(path)

            def execute(command):
                reports = []
                for start in range(0, len(paths), 64):
                    reports.extend(checked([*command, *paths[start:start + 64]], timeout=180).splitlines())
                self.assertEqual(len(reports), len(paths))
                return reports

            expected = execute([ROOT / 'obj/nvm2c_functions_reference'])
            self.assertEqual(expected[0], '1:0:0:1:0')
            self.assertEqual(expected[2], '1:1:1:0:1:4294967295:0:0:00:11:1,0')
            for (label, _, valid), report in zip(cases, expected):
                if valid is not None:
                    with self.subTest(oracle=label): self.assertEqual(report != '0', valid)
            routes = build_routes(ROOT / 'tests/nvm2c_functions_driver.nano', work)
            results = {}
            for label, command in routes:
                results[label] = execute(command)
                for (case, _, _), want, got in zip(cases, expected, results[label]):
                    with self.subTest(route=label, case=case): self.assertEqual(got, want)
            # I retain complete reports before the ordinary temporary cleanup.
            receipt = ROOT / '.test_output/nanolang-functions-results.json'
            receipt.parent.mkdir(exist_ok=True)
            receipt.write_text(json.dumps(dict(cases=[label for label, _, _ in cases],
                explicit=sum(valid is not None for _, _, valid in cases), expected=expected,
                routes=results), indent=2) + '\n')
            print(f'I matched {len(cases)} complete function decodes across {len(routes)} routes; '
                  f'{sum(valid is not None for _, _, valid in cases)} have explicit validity expectations.')


if __name__ == '__main__':
    unittest.main()
