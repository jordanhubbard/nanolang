"""I check written field order through both actual NanoISA producers."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/nanovirt/fixtures/record_literal_written_order.nano'


class RecordLiteralOrder(unittest.TestCase):
    def test_written_order_and_nested_staging(self):
        # I retain commands and products, including any first failure.
        artifacts = Path(tempfile.mkdtemp(prefix='nano-record-literal-order-'))
        commands = []
        for producer in ('nano_virt', 'nanoisa_emit'):
            with self.subTest(producer=producer):
                module = artifacts / (producer + '.nvm')
                for stage, command in (
                    ('emit', [ROOT / 'bin' / producer, SOURCE, '--emit-nvm', '-o', module]),
                    ('verify', [ROOT / 'bin/nano_vm', '--verify-only', module]),
                    ('run', [ROOT / 'bin/nano_vm', module]),
                ):
                    args = list(map(str, command))
                    result = subprocess.run(args, capture_output=True, timeout=45)
                    (artifacts / (producer + '-' + stage + '.stdout')).write_bytes(result.stdout)
                    (artifacts / (producer + '-' + stage + '.stderr')).write_bytes(result.stderr)
                    commands.append({'command': args, 'returncode': result.returncode})
                    (artifacts / 'commands.json').write_text(json.dumps(commands, indent=2))
                    self.assertEqual(result.returncode, 0,
                                     f'{producer} {stage}; retained in {artifacts}\n'
                                     + (result.stdout + result.stderr).decode(errors='replace'))


class NativeLiteralOrder(unittest.TestCase):
    def test_native_aggregates_and_global_initialization(self):
        artifacts = Path(tempfile.mkdtemp(prefix='nano-native-literal-order-'))
        commands = []
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            self.assertTrue((ROOT / 'bin' / compiler).is_file(),
                            'I require fresh bootstrap before this three-producer gate.')
            for fixture in ('record_literal_written_order', 'union_literal_written_order',
                            'global_literal_initialization', 'tuple_array_literal_written_order'):
                with self.subTest(compiler=compiler, fixture=fixture):
                    source = ROOT / 'tests/nanovirt/fixtures' / (fixture + '.nano')
                    binary = artifacts / (compiler + '-' + fixture)
                    for stage, command in (
                        ('compile', [ROOT / 'bin' / compiler, source, '-o', binary]),
                        ('run', [binary]),
                    ):
                        args = list(map(str, command))
                        row = {'command': args, 'stage': stage}
                        commands.append(row)
                        (artifacts / 'commands.json').write_text(json.dumps(commands, indent=2))
                        result = subprocess.run(args, capture_output=True, timeout=90)
                        row['returncode'] = result.returncode
                        prefix = compiler + '-' + fixture + '-' + stage
                        (artifacts / (prefix + '.stdout')).write_bytes(result.stdout)
                        (artifacts / (prefix + '.stderr')).write_bytes(result.stderr)
                        (artifacts / 'commands.json').write_text(json.dumps(commands, indent=2))
                        self.assertEqual(result.returncode, 0,
                                         f'{prefix}; retained in {artifacts}\n'
                                         + (result.stdout + result.stderr).decode(errors='replace'))
