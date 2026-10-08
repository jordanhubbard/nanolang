"""I distinguish NanoISA lowering from backend diagnostics without shifting phases."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from scripts import gen_compiler_schema as schema

ROOT = Path(__file__).resolve().parents[1]
SEED = Path(os.environ.get('NANOLANG_PHASE_CSEED', ROOT / 'bin/nanoc_c'))


class CompilerPhases(unittest.TestCase):
    def checked(self, command, **kwargs):
        result = subprocess.run(list(map(str, command)), cwd=ROOT, capture_output=True,
                                text=True, timeout=180, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def execute(self, source, directory, name):
        path = directory / (name + '.nano')
        module = directory / (name + '.nvm')
        path.write_text(source)
        self.checked([ROOT / 'bin/nano_virt', path, '--emit-nvm', '-o', module])
        self.checked([ROOT / 'bin/nano_vm', module])
        return module

    def test_generated_schema_and_stable_phase_ordinals(self):
        data = schema.load_schema()
        phase = next(item for item in data['nano_enums'] if item['name'] == 'CompilerPhase')
        self.assertEqual(phase['values'], ['PHASE_LEXER', 'PHASE_PARSER', 'PHASE_TYPECHECK',
                                         'PHASE_NANOISA', 'PHASE_RUNTIME', 'PHASE_BACKEND'])
        for path, generated in [(schema.NANO_SCHEMA_OUT, schema.gen_nano_schema(data)),
                                (schema.NANO_AST_OUT, schema.gen_nano_ast(data)),
                                (schema.NANO_CONTRACTS_OUT, schema.gen_contracts(data)),
                                (schema.C_OUT, schema.gen_c(data))]:
            with self.subTest(path=path):
                self.assertEqual(path.read_text(), generated)

    def test_diagnostic_helpers_and_historical_phase_formatters(self):
        prefix = 'import "' + str(ROOT / 'src_nano/compiler/diagnostics.nano') + '"\n'
        source = prefix + '''fn main()->int {
 let loc:CompilerSourceLocation = (diag_location "test.nano" 4 8)
 let lower:CompilerDiagnostic = (diag_nanoisa_error "N001" "I cannot lower this" loc)
 let backend:CompilerDiagnostic = (diag_backend_error "B001" "I cannot link this" loc)
 assert (== lower.phase CompilerPhase.PHASE_NANOISA)
 assert (== backend.phase CompilerPhase.PHASE_BACKEND)
 assert (!= lower.phase backend.phase)
 assert (== lower.code "N001")
 assert (== backend.code "B001")
 assert (== backend.location.column 8)
 return 0
}
shadow main { assert (== (main) 0) }
'''
        with tempfile.TemporaryDirectory(prefix='nano-phase-values-') as tmp:
            work = Path(tmp)
            module = self.execute(source, work, 'helpers')
            native = work / 'helpers.c'
            binary = work / 'helpers'
            self.checked([ROOT / 'bin/nvm2c', module, '-o', native])
            self.checked(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', native, '-o', binary])
            self.checked([binary])
            # I execute each formatter's exact source and adjacent shadow,
            # without importing an obsolete whole compiler as a new product.
            for name in ('compiler_modular.nano', 'nanoc_integrated.nano'):
                text = (ROOT / 'src_nano' / name).read_text()
                start = text.index('fn phase_to_string(')
                end = text.index('fn print_single_diagnostic(', start)
                self.execute(prefix + text[start:end] + 'fn main()->int{return 0}\n', work, name)

    def test_cseed_compiler_failure_reports_backend_phase(self):
        with tempfile.TemporaryDirectory(prefix='nano-phase-failure-') as tmp:
            work = Path(tmp)
            source, output, report = [work / name for name in ('main.nano', 'main', 'diagnostics.json')]
            source.write_text('fn main()->int{return 0}\nshadow main {assert (== (main) 0)}\n')
            result = subprocess.run([str(SEED), str(source), '-o', str(output),
                                     '--llm-diags-json', str(report)], cwd=ROOT,
                                    env={**os.environ, 'NANOLANG_ROOT': str(ROOT), 'NANO_CC': '/usr/bin/false'},
                                    capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            data = json.loads(report.read_text())
            self.assertFalse(data['success'])
            errors = [d for d in data['diagnostics'] if d['severity_name'] == 'error']
            self.assertTrue(errors, data)
            self.assertTrue(all(d['phase_name'] == 'backend' and d['phase'] == 5 for d in errors), data)


if __name__ == '__main__':
    unittest.main()
