"""I check written field order through both actual NanoISA producers."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
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
        self.run_native_fixtures()

    def run_native_fixtures(self, optimization=None):
        artifacts = Path(tempfile.mkdtemp(prefix='nano-native-literal-order-'))
        commands = []
        env = os.environ.copy()
        if optimization is not None:
            real_cc = shlex.split(env.get('NANO_CC') or env.get('CC') or 'cc')
            recorder = artifacts / 'record-cc.py'
            recorder.write_text('import json, os, sys\n'
                'with open(os.environ["NANO_LITERAL_CC_LOG"], "a") as out:\n'
                '    out.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                f'command = {real_cc!r} + sys.argv[1:]\n'
                'os.execvp(command[0], command)\n')
            env['NANO_CC'] = shlex.join([sys.executable, str(recorder)])
            env['NANO_CFLAGS'] = optimization
            (artifacts / 'toolchain.json').write_text(json.dumps({
                'compiler': real_cc, 'optimization': optimization,
                'recorder': str(recorder)}, indent=2))
        for compiler in ('nanoc_c', 'nanoc_stage1', 'nanoc_stage2'):
            self.assertTrue((ROOT / 'bin' / compiler).is_file(),
                            'I require fresh bootstrap before this three-producer gate.')
            for fixture in ('record_literal_written_order', 'union_literal_written_order',
                            'global_literal_initialization', 'tuple_array_literal_written_order'):
                with self.subTest(compiler=compiler, fixture=fixture):
                    source = ROOT / 'tests/nanovirt/fixtures' / (fixture + '.nano')
                    binary = artifacts / (compiler + '-' + fixture)
                    cc_log = artifacts / (compiler + '-' + fixture + '.cc.jsonl')
                    if optimization is not None:
                        env['NANO_LITERAL_CC_LOG'] = str(cc_log)
                    for stage, command in (
                        ('compile', [ROOT / 'bin' / compiler, source, '-o', binary]),
                        ('run', [binary]),
                    ):
                        args = list(map(str, command))
                        row = {'command': args, 'stage': stage, 'optimization': optimization}
                        commands.append(row)
                        (artifacts / 'commands.json').write_text(json.dumps(commands, indent=2))
                        result = subprocess.run(args, capture_output=True, timeout=90, env=env)
                        row['returncode'] = result.returncode
                        prefix = compiler + '-' + fixture + '-' + stage
                        (artifacts / (prefix + '.stdout')).write_bytes(result.stdout)
                        (artifacts / (prefix + '.stderr')).write_bytes(result.stderr)
                        (artifacts / 'commands.json').write_text(json.dumps(commands, indent=2))
                        self.assertEqual(result.returncode, 0,
                                         f'{prefix}; retained in {artifacts}\n'
                                         + (result.stdout + result.stderr).decode(errors='replace'))
                        if stage == 'compile' and optimization is not None:
                            observed = [json.loads(line) for line in cc_log.read_text().splitlines()]
                            final = [argv for argv in observed if any(
                                arg == '-o' and i + 1 < len(argv) and argv[i + 1] == str(binary)
                                for i, arg in enumerate(argv))]
                            self.assertEqual(len(final), 1, (artifacts, observed))
                            flags = [arg for arg in final[0] if arg.startswith('-O')]
                            self.assertTrue(flags, (artifacts, final))
                            self.assertEqual(flags[-1], optimization, (artifacts, final))


class NativeLiteralOptimization(NativeLiteralOrder):
    def test_native_aggregates_and_global_initialization(self):
        for optimization in ('-O0', '-O2'):
            with self.subTest(optimization=optimization):
                self.run_native_fixtures(optimization)
