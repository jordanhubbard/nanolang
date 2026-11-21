#!/usr/bin/env python3
"""I qualify one private version-token candidate and restore the checkout afterward."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.native_process import run_captured
from scripts import ci_source_capture_linux as capture

OUT = ROOT / '.test_output/source-capture-diagnostic/as246-candidate'


def command(argv, label, timeout):
    result = run_captured(argv, cwd=ROOT, timeout=timeout)
    (OUT / (label + '.stdout')).write_bytes(result.stdout)
    (OUT / (label + '.stderr')).write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'I retain {label} refusal with exit {result.returncode}.')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    previous = json.loads((capture.OUT / 'observation.json').read_text())
    if not previous['assembler_version'].splitlines()[0].endswith(') 2.46'):
        raise RuntimeError('I require the actual GNU assembler 2.46 host.')
    source = ROOT / 'src/module_builder.c'
    test = ROOT / 'tests/test_source_snapshots.py'
    original = {path: path.read_bytes() for path in (source, test)}
    record = {'source_pin': previous['commit'], 'product_admission_changed': False,
              'limits': ['I qualify a private checkout delta. Promotion and complete '
                         'source/snippet/platform acceptance remain separate. '
                         'Original unittest temporary products retain their original cleanup.']}
    try:
        text = original[source].decode()
        token = 'static const char *versions[] = {" 2.40", " 2.42"};'
        if text.count(token) != 1:
            raise RuntimeError('I refuse an unexpected production version table.')
        source.write_text(text.replace(token, token.replace('" 2.42"', '" 2.42", " 2.46"')))
        text = original[test].decode()
        old = '"GNU assembler (GNU Binutils for Ubuntu) 2.42\\nCopyright text\\n"]'
        new = old[:-1] + ',\n                    "GNU assembler (GNU Binutils for Ubuntu) 2.46\\nCopyright text\\n"]'
        if text.count(old) != 1:
            raise RuntimeError('I refuse an unexpected version-control fixture.')
        text = text.replace(old, new)
        old = 'rejected = ["GNU assembler (GNU Binutils) 2.41\\n",'
        new = ('rejected = ["GNU assembler (GNU Binutils) 2.46.1\\n", '
               '"GNU assembler (GNU Binutils) 2.460\\n",\n'
               '                    "GNU assembler (GNU Binutils) 2.47\\n", '
               '"GNU assembler (GNU Binutils) 2.46",\n'
               '                    "GNU ld (GNU Binutils) 2.46\\n", '
               '"unrelated banner\\nGNU assembler (GNU Binutils) 2.46\\n",\n'
               '                    "GNU assembler (GNU Binutils) 2.41\\n",')
        if text.count(old) != 1:
            raise RuntimeError('I refuse an unexpected refusal-control fixture.')
        test.write_text(text.replace(old, new))
        patch = subprocess.check_output(['git', 'diff', '--', str(source), str(test)], cwd=ROOT)
        (OUT / 'candidate.patch').write_bytes(patch)
        record['candidate_patch_sha256'] = hashlib.sha256(patch).hexdigest()
        for path in original:
            shutil.copy2(path, OUT / path.name)
        command(['make', '-f', 'Makefile.gnu', '-j2',
                 'obj/test_module_generation_probe', 'bin/nano_as_capture.so'], 'build', 600)
        for name in ('obj/test_module_generation_probe', 'bin/nano_as_capture.so'):
            shutil.copy2(ROOT / name, OUT / Path(name).name)
        runner = '''import json,sys,unittest
suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_source_snapshots')
discovered=suite.countTestCases()
result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(discovered=discovered,ran=result.testsRun,errors=len(result.errors),
 failures=len(result.failures),skipped=[(case.id(),reason) for case,reason in result.skipped])
open(sys.argv[1],'w').write(json.dumps(report,indent=2)+'\\n')
sys.exit(0 if result.wasSuccessful() and discovered==128 and result.testsRun==128 else 1)
'''
        command([sys.executable, '-c', runner, str(OUT / 'tests.json')], 'original-tests', 1500)
        report = json.loads((OUT / 'tests.json').read_text())
        required = ('test_gcc_macro_argument_replays_captured_reads',
                    'test_gcc_missing_capture_helper_preserves_generation_and_recovers',
                    'test_gcc_replay_cleanup_failure_and_tool_selection',
                    'test_gcc_retained_object_reproducibility_and_external_inputs',
                    'test_gcc_assembler_restoration_uses_captured_inputs',
                    'test_gcc_incomplete_assembler_capture_refuses_cold_output',
                    'test_gcc_validation_cleanup_and_cold_failure',
                    'test_ordinary_capture_failure_refuses_live_cold_output',
                    'test_admitted_capture_phase_failure_and_recovery',
                    'test_supported_assembler_version_line')
        if any(case.endswith('.' + name) for case, _ in report['skipped'] for name in required):
            raise RuntimeError('I require the original GNU replay controls to execute.')
        capture.OUT = OUT / 'cold-compiler-support'
        capture.main()
        cold = json.loads((capture.OUT / 'observation.json').read_text())
        if cold.get('returncode') != 0 or cold['supported_version_decision'] != 0:
            raise RuntimeError('I require actual cold compiler support capture and publication.')
        record['private_candidate_qualification_passed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        raise
    finally:
        for path, raw in original.items():
            path.write_bytes(raw)
        record['checkout_sources_restored'] = all(path.read_bytes() == raw for path, raw in original.items())
        (OUT / 'result.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
