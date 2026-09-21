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
